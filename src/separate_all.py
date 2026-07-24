"""
Hybrid DSP-Conditioned Neural Music Source Separation
======================================================
Pipeline (ICASSP submission):

  OPTION 2 — Low-latency streaming via WOLA (Weighted Overlap-Add):
    Audio is partitioned into overlapping 4-second segments (sufficient
    context for HTDemucs' receptive field).  Each segment is processed
    independently and reconstructed via a Hann-windowed overlap-add scheme
    that produces continuous 20 ms output frames with no edge artifacts.

  OPTION 4 — DSP-derived soft masks as Demucs input conditioning:
    Before each segment reaches HTDemucs, three DSP masks are computed in
    the STFT domain and combined into a single soft conditioning mask:
      • REPET background mask  — suppresses the repeating instrumental
        background for the vocal pass; inverted for the instrumental pass.
      • HPSS harmonic/percussive mask — routes tonal content toward vocals,
        transient content toward instrumentals.
      • Mid/Side panning weight — down-weights off-center energy in the
        vocal-conditioned input.
    Each source-specific conditioned waveform is reconstructed via iSTFT
    and fed to HTDemucs instead of the raw mixture.  The neural network
    therefore operates on a signal that has been spectrally pre-shaped
    toward the target source, reducing the ambiguity of its inference task.

Output: separated_2/<song>/vocals.wav  +  instrumentals.wav
        (mono, 16-bit PCM, native sample rate, chunked at 20 ms)
"""

import os
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ── constants ─────────────────────────────────────────────────────────────────
CHUNK_MS      = 20          # output frame size in milliseconds
SEGMENT_SEC   = 4.0         # Demucs context window (seconds)
OVERLAP_RATIO = 0.5         # 50 % overlap between segments → smooth WOLA
N_FFT         = 2048
HOP_LENGTH    = 512
REPET_MIN_S   = 0.8         # REPET: shortest repeating period to search (s)
REPET_MAX_S   = 8.0         # REPET: longest repeating period to search (s)
HPSS_MARGIN   = 2.0         # HPSS aggressiveness
DEVICE        = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load_stereo(path: Path):
    """Load MP3 as float32 stereo at native sample rate."""
    y, sr = librosa.load(str(path), sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32), sr   # (2, n_samples)


def save_wav(path: Path, wave: np.ndarray, sr: int):
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    sf.write(str(path), (wave * 32767).astype(np.int16), sr)


# ── DSP helpers ───────────────────────────────────────────────────────────────

def do_stft(x: np.ndarray) -> np.ndarray:
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH)


def do_istft(S: np.ndarray, length: int) -> np.ndarray:
    return librosa.istft(S, hop_length=HOP_LENGTH, length=length)


def repet_background_mask(mag: np.ndarray, sr: int):
    """
    REPET background mask in [0,1]: high where energy is rhythmically repeating.
    Returns (bg_mask, fg_mask) both shape (n_bins, n_frames).
    """
    n_bins, n_frames = mag.shape
    fps      = sr / HOP_LENGTH
    min_lag  = max(1, int(np.round(REPET_MIN_S * fps)))
    max_lag  = min(n_frames // 2, int(np.round(REPET_MAX_S * fps)))

    # Beat spectrum: autocorrelation of spectrogram columns
    beat_spec = np.array([
        np.sum(mag[:, :n_frames - lag] * mag[:, lag:])
        for lag in range(min_lag, max_lag + 1)
    ])
    period = int(np.argmax(beat_spec) + min_lag)

    # Build repeating background model: per-phase median
    background = np.zeros_like(mag)
    for phase in range(period):
        idx = np.arange(phase, n_frames, period)
        if len(idx):
            background[:, idx] = np.median(mag[:, idx], axis=1, keepdims=True)

    eps = 1e-8
    bg_p = background ** 2
    fg_p = (np.maximum(mag - background, 0.0)) ** 2
    bg_mask = bg_p / (bg_p + fg_p + eps)
    fg_mask = 1.0 - bg_mask
    return bg_mask, fg_mask


def hpss_masks(mag: np.ndarray):
    """Wiener-style harmonic and percussive masks from HPSS, shape (n_bins, n_frames)."""
    H, P = librosa.decompose.hpss(mag, margin=HPSS_MARGIN)
    eps  = 1e-8
    denom = H + P + eps
    return H / denom, P / denom   # h_mask, p_mask


def midside_center_weight(S_left: np.ndarray, S_right: np.ndarray):
    """Center-panning weight: high where L ≈ R (center-panned vocals)."""
    eps    = 1e-8
    M_mid  = np.abs(S_left + S_right) / 2.0
    M_side = np.abs(S_left - S_right) / 2.0
    return M_mid / (M_mid + M_side + eps)   # (n_bins, n_frames)


def dsp_condition(segment: np.ndarray, sr: int):
    """
    DSP Input Conditioning (Option 4).

    Given a raw stereo segment (2, n_samples), compute REPET + HPSS + Mid/Side
    masks and return two source-conditioned waveforms:
      vocal_conditioned    — spectrally shaped toward vocals
      inst_conditioned     — spectrally shaped toward instrumentals

    Each conditioned waveform is what HTDemucs will receive instead of the
    raw mixture, reducing inference ambiguity.
    """
    n_samples = segment.shape[1]
    left, right = segment[0], segment[1]
    mid  = (left + right) / 2.0
    side = (left - right) / 2.0

    S_left  = do_stft(left)
    S_right = do_stft(right)
    S_mid   = do_stft(mid)
    S_side  = do_stft(side)

    M_mid  = np.abs(S_mid)
    M_side = np.abs(S_side)
    ph_mid  = np.exp(1j * np.angle(S_mid))
    ph_side = np.exp(1j * np.angle(S_side))

    # DSP masks
    repet_bg, repet_fg = repet_background_mask(M_mid, sr)
    h_mask, p_mask     = hpss_masks(M_mid)
    center_w           = midside_center_weight(S_left, S_right)

    # Vocal conditioning mask: foreground × harmonic × center-panned
    vocal_mask = repet_fg * h_mask * center_w
    # Instrumental conditioning mask: background × (harmonic + percussive) × side-weighted
    side_w     = 1.0 - center_w
    inst_mask  = (repet_bg * h_mask + p_mask) * (side_w + 0.3)
    inst_mask  = np.clip(inst_mask, 0.0, 1.0)

    # Reconstruct conditioned waveforms (mono → promote to stereo for Demucs)
    vocal_wave = do_istft(vocal_mask * M_mid * ph_mid,  length=n_samples)
    inst_wave  = do_istft(inst_mask  * M_mid * ph_side, length=n_samples)

    # Blend 70 % conditioned + 30 % original so Demucs still has full-mix context
    vocal_stereo = 0.7 * np.stack([vocal_wave, vocal_wave]) + 0.3 * segment
    inst_stereo  = 0.7 * np.stack([inst_wave,  inst_wave])  + 0.3 * segment

    return vocal_stereo.astype(np.float32), inst_stereo.astype(np.float32)


# ── WOLA streaming engine (Option 2) ─────────────────────────────────────────

class WOLAStream:
    """
    Weighted Overlap-Add streaming reconstructor.

    Accepts processed segments of length `seg_len` with `hop_len` spacing,
    accumulates them with a Hann window, and emits complete 20 ms frames.
    """
    def __init__(self, seg_len: int, hop_len: int, chunk_len: int):
        self.seg_len   = seg_len
        self.hop_len   = hop_len
        self.chunk_len = chunk_len
        self.window    = np.hanning(seg_len).astype(np.float32)
        self.out_buf   = np.zeros(seg_len * 4, dtype=np.float32)
        self.w_buf     = np.zeros(seg_len * 4, dtype=np.float32)
        self.write_pos = 0
        self.read_pos  = 0
        self.chunks    = []

    def add_segment(self, seg: np.ndarray):
        """Add one overlap-add segment."""
        end = self.write_pos + self.seg_len
        if end > len(self.out_buf):
            # Grow buffers
            pad = end - len(self.out_buf)
            self.out_buf = np.concatenate([self.out_buf, np.zeros(pad + self.seg_len)])
            self.w_buf   = np.concatenate([self.w_buf,   np.zeros(pad + self.seg_len)])
        self.out_buf[self.write_pos:end] += seg * self.window
        self.w_buf  [self.write_pos:end] += self.window
        self.write_pos += self.hop_len

        # Drain completed 20 ms chunks
        while self.read_pos + self.chunk_len <= self.write_pos - self.hop_len:
            start, stop = self.read_pos, self.read_pos + self.chunk_len
            w = self.w_buf[start:stop]
            w = np.where(w > 1e-8, w, 1.0)
            frame = self.out_buf[start:stop] / w
            self.chunks.append(frame.copy())
            self.read_pos += self.chunk_len

    def flush(self) -> np.ndarray:
        """Return all accumulated output as a single array."""
        return np.concatenate(self.chunks) if self.chunks else np.array([], dtype=np.float32)


# ── Demucs runner ─────────────────────────────────────────────────────────────

def load_demucs(model_name="htdemucs"):
    print(f"  Loading HTDemucs model on {DEVICE}...")
    model = get_model(model_name)
    model.to(DEVICE)
    model.eval()
    return model


def run_demucs(model, stereo_np: np.ndarray, sr: int):
    """
    Run HTDemucs on a stereo numpy array (2, n_samples).
    Returns dict: source_name → np.ndarray (2, n_samples).
    """
    target_sr = model.samplerate
    if sr != target_sr:
        stereo_np = librosa.resample(stereo_np, orig_sr=sr, target_sr=target_sr)

    tensor = torch.from_numpy(stereo_np).unsqueeze(0).to(DEVICE)  # (1, 2, n)
    with torch.no_grad():
        out = apply_model(model, tensor, device=DEVICE, shifts=1, split=True, overlap=0.25)
    # out: (1, n_sources, 2, n_samples)
    sources = {}
    for i, name in enumerate(model.sources):
        src = out[0, i].cpu().numpy()  # (2, n_samples)
        if sr != target_sr:
            src = librosa.resample(src, orig_sr=target_sr, target_sr=sr)
        sources[name] = src
    return sources


# ── per-song separation ───────────────────────────────────────────────────────

def separate(input_path: Path, output_dir: Path, model):
    song    = input_path.stem
    out     = output_dir / song
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {song}")
    print(f"{'='*60}")

    print("  [1/4] Loading audio...")
    audio, sr = load_stereo(input_path)
    n_samples  = audio.shape[1]

    chunk_len = int(sr * CHUNK_MS / 1000)           # 20 ms in samples
    seg_len   = int(sr * SEGMENT_SEC)               # Demucs context window
    hop_len   = int(seg_len * (1 - OVERLAP_RATIO))  # 50 % overlap hop

    # Pad so the last segment is full
    pad       = seg_len - (n_samples % hop_len or hop_len)
    audio_pad = np.pad(audio, ((0, 0), (0, pad)))
    n_pad     = audio_pad.shape[1]

    # WOLA buffers for vocal and instrumental streams
    stream_v = WOLAStream(seg_len, hop_len, chunk_len)
    stream_i = WOLAStream(seg_len, hop_len, chunk_len)

    starts   = np.arange(0, n_pad - seg_len + 1, hop_len)
    n_segs   = len(starts)
    t_dsp    = 0.0
    t_demucs = 0.0

    print(f"  [2/4] DSP conditioning + HTDemucs ({n_segs} segments × {SEGMENT_SEC}s, {OVERLAP_RATIO*100:.0f}% overlap)...")

    for seg_idx, start in enumerate(starts):
        seg = audio_pad[:, start:start + seg_len]

        # ── Option 4: DSP input conditioning ─────────────────────────────
        t0 = time.perf_counter()
        vocal_in, inst_in = dsp_condition(seg, sr)
        t_dsp += time.perf_counter() - t0

        # ── Option 2 step A: run Demucs on each conditioned input ────────
        t0 = time.perf_counter()
        src_v = run_demucs(model, vocal_in, sr)
        src_i = run_demucs(model, inst_in,  sr)
        t_demucs += time.perf_counter() - t0

        # Vocal output: "vocals" source from the vocal-conditioned run
        # Instrumental: sum of drums+bass+other from the inst-conditioned run
        vocal_seg = src_v["vocals"].mean(axis=0)     # mono
        inst_seg  = (
            src_i.get("drums",  np.zeros(seg_len)).mean(axis=0) +
            src_i.get("bass",   np.zeros(seg_len)).mean(axis=0) +
            src_i.get("other",  np.zeros(seg_len)).mean(axis=0)
        )

        # ── Option 2 step B: push into WOLA stream → emits 20 ms frames ─
        stream_v.add_segment(vocal_seg)
        stream_i.add_segment(inst_seg)

        if (seg_idx + 1) % 5 == 0 or seg_idx == n_segs - 1:
            print(f"    segment {seg_idx+1}/{n_segs}  |  "
                  f"DSP {t_dsp:.1f}s  Demucs {t_demucs:.1f}s")

    print(f"  [3/4] Assembling 20 ms output frames...")
    vocal_out = stream_v.flush()[:n_samples]
    inst_out  = stream_i.flush()[:n_samples]

    print(f"  [4/4] Saving...")
    save_wav(out / "vocals.wav",        vocal_out, sr)
    save_wav(out / "instrumentals.wav", inst_out,  sr)

    total = t_dsp + t_demucs
    print(f"  Done — DSP {t_dsp:.1f}s | Demucs {t_demucs:.1f}s | Total {total:.1f}s")
    print(f"  Output: {out}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    input_dir  = Path("input")
    output_dir = Path("separated_2")

    mp3_files = sorted(input_dir.glob("*.mp3"))
    if not mp3_files:
        print("No MP3 files found in input/")
        return

    print("=" * 60)
    print("  HYBRID DSP-CONDITIONED NEURAL SOURCE SEPARATOR")
    print("  Option 2: WOLA 20ms streaming  |  Option 4: DSP conditioning")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    print(f"  Files: {[f.name for f in mp3_files]}")

    model = load_demucs()

    for f in mp3_files:
        separate(f, output_dir, model)

    print("\n" + "=" * 60)
    print(f"  All done. Outputs in {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
