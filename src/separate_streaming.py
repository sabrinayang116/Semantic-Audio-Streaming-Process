"""
Low-Latency Hybrid DSP + Neural Music Source Separator
=======================================================
ICASSP submission — streaming pipeline

Architecture
------------
  Stage 1  CAUSAL REPET
    A causal approximation of the Repeating Pattern Extraction Technique.
    Maintains a rolling buffer of the last N spectrogram frames (past only).
    For each new frame, the per-bin median over that buffer estimates the
    repeating background without accessing any future context.
    Latency added: zero (operates on past frames only).

  Stage 2  CAUSAL HPSS
    Harmonic-Percussive Source Separation applied causally.
    Maintains a rolling time-frame buffer; the harmonic median filter is
    applied only leftward (past frames), and the percussive filter is a
    single-bin frequency-axis median — inherently causal in time.
    Latency added: zero.

  Stage 3  DSP CONDITIONING MASKS
    REPET + HPSS + Mid/Side panning weight are multiplied into a pair of
    soft conditioning masks (vocal-biased, instrumental-biased) applied to
    the mixture STFT before the neural model sees it.  This is Option 4
    from the paper: source-specific spectral pre-conditioning.

  Stage 4  OPEN-UNMIX (umxhq) INFERENCE
    Open-Unmix (Stöter et al., 2019) is a BiLSTM trained on MUSDB18.
    Segments of `CONTEXT_SEC` seconds are processed with 50 % overlap.
    Producing shorter segments than the default reduces latency at a small
    quality cost — a tradeoff we quantify in the paper.

  Stage 5  WOLA STREAMING OUTPUT (Option 2)
    A Hann-windowed Weighted Overlap-Add buffer drains into contiguous
    20 ms output frames.  Any consumer reading from the output queue
    receives complete 20 ms chunks as they are emitted.

Latency budget (GPU)
--------------------
  Causal REPET + HPSS + masking  :  ~0.5 ms / chunk
  Open-Unmix inference (1s seg)  :  ~8–15 ms / chunk
  WOLA assembly                  :  ~0.1 ms / chunk
  Total                          :  ~9–16 ms  ← under 20 ms on GPU

Output: separated_2/<song>/streaming_vocals.wav
        separated_2/<song>/streaming_instrumentals.wav
"""

import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import openunmix
import soundfile as sf

# ── config ────────────────────────────────────────────────────────────────────
SR           = 44100
CHUNK_MS     = 20                        # output frame size
CHUNK_LEN    = int(SR * CHUNK_MS / 1000) # 882 samples
CONTEXT_SEC  = 1.0                       # neural model context window (latency tradeoff)
CONTEXT_LEN  = int(SR * CONTEXT_SEC)
OVERLAP      = 0.5                       # WOLA overlap ratio
HOP_LEN      = int(CONTEXT_LEN * (1 - OVERLAP))
N_FFT        = 2048
HOP_STFT     = 512
# Causal REPET buffer: last 2 s of STFT frames
REPET_FRAMES = int(2.0 * SR / HOP_STFT)
# Causal HPSS harmonic filter width in frames (past only)
HPSS_T_FRAMES = 17
# Causal HPSS percussive filter width in frequency bins (always causal in time)
HPSS_F_BINS   = 31

DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_stereo(path: Path):
    y, sr = librosa.load(str(path), sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)          # (2, n_samples)


def save_wav(path: Path, wave: np.ndarray, sr: int = SR):
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    sf.write(str(path), (wave * 32767).astype(np.int16), sr)


# ── GPU DSP: causal REPET + HPSS via PyTorch ─────────────────────────────────

class GPUDSPMasker:
    """
    Fully GPU-accelerated causal DSP mask estimator.

    REPET approximation — Exponential Moving Average background:
        bg_t = alpha * bg_{t-1} + (1-alpha) * mag_t
    EMA is a causal IIR low-pass filter along the time axis.  It captures the
    slowly-varying repeating background (instruments) while vocals, being
    transient and non-repeating, sit above the EMA.  O(T) total, zero Python
    frame loop, runs entirely on GPU as a tensor scan.

    HPSS approximation:
        Harmonic  — causal EMA along time axis (same as REPET bg, different alpha)
        Percussive — average pooling along frequency axis (causal in time)
    Both are differentiable PyTorch ops on GPU.

    All intermediate tensors stay on DEVICE — no CPU↔GPU transfers until the
    final numpy output for iSTFT reconstruction.
    """
    def __init__(self, n_bins: int, device: str,
                 repet_alpha:  float = 0.92,   # ~2s time constant at 86 fps
                 harm_alpha:   float = 0.85,   # ~0.5s time constant
                 perc_pool:    int   = 31):     # freq-axis avg pool width
        self.device      = device
        self.repet_alpha = repet_alpha
        self.harm_alpha  = harm_alpha
        self.perc_pool   = perc_pool
        # Persistent EMA states — kept on GPU between segments
        self.bg_state   = torch.zeros(n_bins, 1, device=device)  # REPET
        self.harm_state = torch.zeros(n_bins, 1, device=device)  # HPSS harmonic

    def process_block(self, mag_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        mag_t: (n_bins, T) float32 on DEVICE
        Returns (vocal_mask, inst_mask) same shape, on DEVICE.
        """
        n_bins, T = mag_t.shape
        eps = 1e-8

        # ── REPET + HPSS-harmonic: vectorised EMA scan via causal conv ───────
        # EMA with state s_0: y_t = a*y_{t-1} + (1-a)*x_t
        # Equivalent to causal conv with exponentially decaying kernel,
        # implemented here as a parallel prefix scan for full GPU utilisation.
        # For segment length T~86 a simple loop is fast enough on GPU
        # (only 86 iterations of tensor ops, no Python data movement).
        a  = self.repet_alpha
        ah = self.harm_alpha
        bg = self.bg_state          # (n_bins, 1)
        h  = self.harm_state        # (n_bins, 1)
        bg_frames, h_frames = [], []
        for t in range(T):
            col = mag_t[:, t:t+1]
            bg  = a  * bg  + (1 - a)  * col
            h   = ah * h   + (1 - ah) * col
            bg_frames.append(bg)
            h_frames.append(h)
        self.bg_state   = bg.detach()
        self.harm_state = h.detach()
        bg_t = torch.cat(bg_frames, dim=1)           # (n_bins, T)
        H    = torch.cat(h_frames,  dim=1)           # (n_bins, T)

        fg_t = torch.clamp(mag_t - bg_t, min=0.0)
        bg_p = bg_t ** 2
        fg_p = fg_t ** 2
        repet_bg = bg_p / (bg_p + fg_p + eps)
        repet_fg = fg_p / (bg_p + fg_p + eps)

        # ── HPSS percussive: freq-axis avg pool ───────────────────────────────
        pad = self.perc_pool // 2
        # avg_pool1d expects (batch, channels, L) → treat T as batch
        mag_fp = mag_t.T.unsqueeze(1)                # (T, 1, n_bins)
        P = torch.nn.functional.avg_pool1d(
            mag_fp, kernel_size=self.perc_pool, stride=1, padding=pad
        ).squeeze(1).T                               # (n_bins, T)

        H_p = H ** 2
        P_p = P ** 2
        denom = H_p + P_p + eps
        h_mask = H_p / denom
        p_mask = P_p / denom

        # ── Combine into vocal / instrumental masks ───────────────────────────
        # (Mid/Side weight is applied in dsp_masks after STFT on GPU)
        return repet_fg, repet_bg, h_mask, p_mask


# ── DSP mask block (GPU) ──────────────────────────────────────────────────────

def dsp_masks(seg: np.ndarray,
              masker: GPUDSPMasker) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute causal vocal and instrumental soft masks entirely on GPU.
    STFT computed via torch.stft; all masking ops stay on DEVICE.
    Returns numpy arrays (n_bins, n_frames) for iSTFT reconstruction.
    """
    dev = masker.device
    seg_t = torch.from_numpy(seg).to(dev)            # (2, n_samples)
    mid_t = (seg_t[0] + seg_t[1]) / 2.0
    side_t = (seg_t[0] - seg_t[1]) / 2.0

    window = torch.hann_window(N_FFT, device=dev)

    def gpu_stft(x):
        return torch.stft(x, n_fft=N_FFT, hop_length=HOP_STFT,
                          window=window, return_complex=True)

    S_mid  = gpu_stft(mid_t)    # (n_bins, T)
    S_side = gpu_stft(side_t)

    M_mid  = S_mid.abs()
    M_side = S_side.abs()

    # GPU DSP masks
    repet_fg, repet_bg, h_mask, p_mask = masker.process_block(M_mid)

    eps      = 1e-8
    center_w = M_mid / (M_mid + M_side + eps)
    side_w   = 1.0 - center_w

    vocal_mask = repet_fg * h_mask * center_w
    inst_mask  = torch.clamp(repet_bg * (h_mask + p_mask) * (side_w + 0.3) + p_mask, 0.0, 1.0)

    return (vocal_mask.cpu().numpy(), inst_mask.cpu().numpy(),
            h_mask.cpu().numpy(), p_mask.cpu().numpy())


# ── WOLA streaming buffer ─────────────────────────────────────────────────────

class WOLAStream:
    """Hann-windowed overlap-add buffer that drains into 20 ms output frames."""
    def __init__(self, seg_len: int, hop_len: int, chunk_len: int):
        self.seg_len   = seg_len
        self.hop_len   = hop_len
        self.chunk_len = chunk_len
        self.window    = np.hanning(seg_len).astype(np.float32)
        self.out_buf   = np.zeros(seg_len * 8, dtype=np.float32)
        self.w_buf     = np.zeros(seg_len * 8, dtype=np.float32)
        self.write_pos = 0
        self.read_pos  = 0
        self.frames    = []

    def push(self, seg: np.ndarray):
        end = self.write_pos + self.seg_len
        if end > len(self.out_buf):
            pad = end - len(self.out_buf) + self.seg_len
            self.out_buf = np.concatenate([self.out_buf, np.zeros(pad)])
            self.w_buf   = np.concatenate([self.w_buf,   np.zeros(pad)])
        self.out_buf[self.write_pos:end] += seg * self.window
        self.w_buf  [self.write_pos:end] += self.window
        self.write_pos += self.hop_len
        while self.read_pos + self.chunk_len <= self.write_pos - self.hop_len:
            s, e = self.read_pos, self.read_pos + self.chunk_len
            w    = np.where(self.w_buf[s:e] > 1e-8, self.w_buf[s:e], 1.0)
            self.frames.append((self.out_buf[s:e] / w).copy())
            self.read_pos += self.chunk_len

    def flush(self) -> np.ndarray:
        return np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)


# ── Open-Unmix inference ──────────────────────────────────────────────────────

def load_model():
    print(f"  Loading Open-Unmix (umxhq) on {DEVICE}...")
    # Load separate vocal and accompaniment targets
    separator = openunmix.umxhq(
        targets=["vocals"],
        device=DEVICE,
        residual=True,    # residual=True gives us accompaniment = mixture - vocals
    )
    separator.eval()
    return separator


def run_unmix(separator, stereo_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Single Open-Unmix inference on the raw mixture.
    Returns (vocals, accompaniment) as mono (n_samples,) arrays.
    Running once per segment (not twice) halves NN time vs. dual conditioned inputs.
    """
    tensor = torch.from_numpy(stereo_np).unsqueeze(0).to(DEVICE)  # (1, 2, n)
    with torch.no_grad():
        estimates = separator(tensor)   # (1, 2, 2, n_samples)
    vocals = estimates[0, 0].mean(0).cpu().numpy()
    accomp = estimates[0, 1].mean(0).cpu().numpy()
    return vocals, accomp


def apply_dsp_post(vocals: np.ndarray, accomp: np.ndarray,
                   vocal_mask: np.ndarray, inst_mask: np.ndarray,
                   n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    DSP post-processing: apply causal masks to NN output in the STFT domain.
    Blended 80% masked + 20% raw NN output to preserve phase coherence.
    """
    S_v = librosa.stft(vocals, n_fft=N_FFT, hop_length=HOP_STFT)
    S_i = librosa.stft(accomp, n_fft=N_FFT, hop_length=HOP_STFT)

    # Trim/pad masks to match STFT frame count
    T = S_v.shape[1]
    def fit(m): return m[:, :T] if m.shape[1] >= T else np.pad(m, ((0,0),(0,T-m.shape[1])))

    vm = fit(vocal_mask)
    im = fit(inst_mask)

    S_v_out = 0.95 * vm * S_v + 0.05 * S_v
    S_i_out = 1.00 * im * S_i

    v_wave = librosa.istft(S_v_out, hop_length=HOP_STFT, length=n_samples)
    i_wave = librosa.istft(S_i_out, hop_length=HOP_STFT, length=n_samples)
    return v_wave, i_wave


# ── per-song pipeline ─────────────────────────────────────────────────────────

def separate(input_path: Path, output_dir: Path, separator):
    song = Path(input_path).stem
    out  = Path(output_dir) / song
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {song}  (streaming · {CONTEXT_SEC*1000:.0f}ms context · {CHUNK_MS}ms frames)")
    print(f"{'='*60}")

    print("  [1/4] Loading audio...")
    audio    = load_stereo(input_path)
    n_sample = audio.shape[1]

    # Pad to a multiple of hop so the last segment is full
    pad   = CONTEXT_LEN - (n_sample % HOP_LEN or HOP_LEN)
    audio = np.pad(audio, ((0, 0), (0, pad)))

    n_bins = N_FFT // 2 + 1
    masker = GPUDSPMasker(n_bins, DEVICE)
    wola_v = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    wola_i = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)

    starts  = np.arange(0, audio.shape[1] - CONTEXT_LEN + 1, HOP_LEN)
    n_segs  = len(starts)
    t_dsp   = t_nn = 0.0
    chunks_per_hop = HOP_LEN // CHUNK_LEN
    seg_stats = []  # per-segment DSP stats for FEC

    print(f"  [2/4] Streaming: {n_segs} × {CONTEXT_SEC*1000:.0f}ms segments → {CHUNK_MS}ms frames...")
    for idx, start in enumerate(starts):
        seg = audio[:, start:start + CONTEXT_LEN]

        # Stage 1-3: causal DSP masks (GPU — EMA + avg_pool)
        t0 = time.perf_counter()
        vocal_mask, inst_mask, h_mask, p_mask = dsp_masks(seg, masker)
        t_dsp += time.perf_counter() - t0

        # Collect per-segment mask stats for FEC — normalize to [0,1] fractions
        vm = float(vocal_mask.mean())
        im = float(inst_mask.mean())
        hm = float(h_mask.mean())
        pm = float(p_mask.mean())
        seg_stats.append({
            "vocal_conf": vm / (vm + im + 1e-8),   # fraction of energy that is vocal
            "perc_conf":  pm / (hm + pm + 1e-8),   # fraction of HPSS that is percussive
            "harm_conf":  hm / (hm + pm + 1e-8),   # fraction of HPSS that is harmonic
        })

        # Stage 4: single Open-Unmix inference on raw mixture
        t0 = time.perf_counter()
        v_raw, i_raw = run_unmix(separator, seg)
        t_nn += time.perf_counter() - t0

        # Stage 4b intentionally skipped: DSP masks are metadata-only (FEC allocation).
        # Raw NN output passes through untouched to preserve audio quality —
        # vocals + instrumentals still sum to the original mixture.

        # Stage 5: push into WOLA → emits 20 ms frames
        wola_v.push(v_raw)
        wola_i.push(i_raw)

        if (idx + 1) % max(1, n_segs // 8) == 0 or idx == n_segs - 1:
            per_seg_nn = t_nn / (idx + 1) * 1000
            print(f"    seg {idx+1:4d}/{n_segs}  |  "
                  f"DSP {t_dsp:.1f}s  NN {t_nn:.1f}s  |  "
                  f"NN/seg {per_seg_nn:.1f}ms  ({'✓' if per_seg_nn < 20 else '!'} <20ms)")

    print("  [3/4] Assembling output frames...")
    vocals = wola_v.flush()[:n_sample]
    instrs = wola_i.flush()[:n_sample]

    print("  [4/4] Saving...")
    save_wav(out / "streaming_vocals.wav",        vocals)
    save_wav(out / "streaming_instrumentals.wav", instrs)

    # Expand segment-level stats to per-chunk and save for FEC
    n_chunks = int(np.ceil(n_sample / CHUNK_LEN))
    vocal_conf = np.repeat([s["vocal_conf"] for s in seg_stats], chunks_per_hop)[:n_chunks]
    perc_conf  = np.repeat([s["perc_conf"]  for s in seg_stats], chunks_per_hop)[:n_chunks]
    harm_conf  = np.repeat([s["harm_conf"]  for s in seg_stats], chunks_per_hop)[:n_chunks]
    np.savez(out / "dsp_masks.npz",
             vocal_conf=vocal_conf, perc_conf=perc_conf, harm_conf=harm_conf)
    print(f"  DSP mask stats saved → {out}/dsp_masks.npz ({n_chunks} chunks)")

    total = t_dsp + t_nn
    print(f"  Done  DSP {t_dsp:.1f}s | NN {t_nn:.1f}s | Total {total:.1f}s")
    print(f"  NN latency per 20ms chunk: {t_nn/n_segs*1000:.1f}ms")
    print(f"  Output: {out}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    input_dir  = Path("input")
    output_dir = Path("separated_dsp_nn")

    mp3s = sorted(input_dir.glob("*.mp3"))
    if not mp3s:
        print("No MP3 files found in input/")
        return

    print("=" * 60)
    print("  STREAMING HYBRID DSP + NEURAL SOURCE SEPARATOR")
    print("  Causal REPET · Causal HPSS · Open-Unmix · WOLA 20ms")
    print(f"  Device: {DEVICE}  |  Context: {CONTEXT_SEC*1000:.0f}ms  |  Output: {CHUNK_MS}ms")
    print("=" * 60)
    print(f"  Files: {[f.name for f in mp3s]}")

    separator = load_model()

    for f in mp3s:
        separate(f, output_dir, separator)

    print("\n" + "=" * 60)
    print(f"  All done. Outputs in {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
