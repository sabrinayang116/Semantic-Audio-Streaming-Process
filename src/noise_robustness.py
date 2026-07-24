"""
Noise Robustness Test
=====================
Tests how robust the DSP+NN separation pipeline is to input noise.

Pipeline per noise level:
  1. Add white Gaussian noise to the original MP3 at a given SNR
  2. Run DSP+NN separation on the noisy input
  3. Run FEC simulation (EEP vs UEP) on the separated outputs
  4. Compare recovered audio against the clean reference separation
  5. Report quality (SNR 0-100) across noise levels and loss rates

Noise levels tested: 30dB (nearly clean), 20dB, 10dB, 0dB (very noisy)
Loss rates tested:   1%, 5%, 10%, 20%

Output: noise_robustness/<song>/snr_<X>db/<EEP|UEP>/vocals.wav + instrumentals.wav
"""

import os
import time
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import openunmix
import soundfile as sf
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SR            = 44100
CHUNK_MS      = 20
CHUNK_LEN     = int(SR * CHUNK_MS / 1000)
CONTEXT_SEC   = 1.0
CONTEXT_LEN   = int(SR * CONTEXT_SEC)
OVERLAP       = 0.5
HOP_LEN       = int(CONTEXT_LEN * (1 - OVERLAP))
N_FFT         = 2048
HOP_STFT      = 512
REPET_FRAMES  = int(2.0 * SR / HOP_STFT)
NOISE_LEVELS  = [30, 20, 10, 0]   # input SNR in dB
LOSS_RATES    = [0.01, 0.05, 0.10, 0.20]
REDUND_BUDGET = 3
MIN_REDUND    = 1
VOCAL_BIAS    = 0.10
PERC_WEIGHT   = 0.30
SILENCE_THRESH = 1e-4

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ── noise injection ────────────────────────────────────────────────────────────

def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to audio at a given SNR level."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.randn(*audio.shape).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def save_wav(path: Path, wave: np.ndarray):
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    sf.write(str(path), (wave * 32767).astype(np.int16), SR)


def load_wav(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray):
    n        = len(audio)
    n_chunks = int(np.ceil(n / CHUNK_LEN))
    padded   = np.zeros(n_chunks * CHUNK_LEN, dtype=np.float32)
    padded[:n] = audio
    return padded.reshape(n_chunks, CHUNK_LEN), n


# ── DSP masker (same as separate_streaming.py) ────────────────────────────────

class GPUDSPMasker:
    def __init__(self, n_bins, device, repet_alpha=0.92, harm_alpha=0.85, perc_pool=31):
        self.device      = device
        self.repet_alpha = repet_alpha
        self.harm_alpha  = harm_alpha
        self.perc_pool   = perc_pool
        self.bg_state    = torch.zeros(n_bins, 1, device=device)
        self.harm_state  = torch.zeros(n_bins, 1, device=device)

    def reset(self):
        self.bg_state.zero_()
        self.harm_state.zero_()

    def process_block(self, mag_t):
        n_bins, T = mag_t.shape
        eps = 1e-8
        a, ah = self.repet_alpha, self.harm_alpha
        bg, h = self.bg_state, self.harm_state
        bg_frames, h_frames = [], []
        for t in range(T):
            col = mag_t[:, t:t+1]
            bg  = a  * bg  + (1 - a)  * col
            h   = ah * h   + (1 - ah) * col
            bg_frames.append(bg)
            h_frames.append(h)
        self.bg_state   = bg.detach()
        self.harm_state = h.detach()
        bg_t = torch.cat(bg_frames, dim=1)
        H    = torch.cat(h_frames,  dim=1)
        fg_t = torch.clamp(mag_t - bg_t, min=0.0)
        bg_p, fg_p = bg_t ** 2, fg_t ** 2
        repet_fg = fg_p / (bg_p + fg_p + eps)
        repet_bg = bg_p / (bg_p + fg_p + eps)
        pad = self.perc_pool // 2
        mag_fp = mag_t.T.unsqueeze(1)
        P = F.avg_pool1d(mag_fp, kernel_size=self.perc_pool, stride=1,
                         padding=pad).squeeze(1).T
        H_p, P_p = H ** 2, P ** 2
        denom = H_p + P_p + eps
        h_mask = H_p / denom
        p_mask = P_p / denom
        return repet_fg, repet_bg, h_mask, p_mask


def dsp_masks(seg, masker):
    dev   = masker.device
    seg_t = torch.from_numpy(seg).to(dev)
    mid_t  = (seg_t[0] + seg_t[1]) / 2.0
    side_t = (seg_t[0] - seg_t[1]) / 2.0
    window = torch.hann_window(N_FFT, device=dev)
    def gpu_stft(x):
        return torch.stft(x, n_fft=N_FFT, hop_length=HOP_STFT,
                          window=window, return_complex=True)
    S_mid  = gpu_stft(mid_t)
    S_side = gpu_stft(side_t)
    M_mid  = S_mid.abs()
    M_side = S_side.abs()
    repet_fg, repet_bg, h_mask, p_mask = masker.process_block(M_mid)
    eps      = 1e-8
    center_w = M_mid / (M_mid + M_side + eps)
    side_w   = 1.0 - center_w
    vocal_mask = repet_fg * h_mask * center_w
    inst_mask  = torch.clamp(repet_bg * (h_mask + p_mask) * (side_w + 0.3) + p_mask, 0.0, 1.0)
    return (vocal_mask.cpu().numpy(), inst_mask.cpu().numpy(),
            h_mask.cpu().numpy(), p_mask.cpu().numpy())


class WOLAStream:
    def __init__(self, seg_len, hop_len, chunk_len):
        self.seg_len   = seg_len
        self.hop_len   = hop_len
        self.chunk_len = chunk_len
        self.window    = np.hanning(seg_len).astype(np.float32)
        self.out_buf   = np.zeros(seg_len * 8, dtype=np.float32)
        self.w_buf     = np.zeros(seg_len * 8, dtype=np.float32)
        self.write_pos = 0
        self.read_pos  = 0
        self.frames    = []

    def push(self, seg):
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

    def flush(self):
        return np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)


def apply_dsp_post(vocals, accomp, vocal_mask, inst_mask, n_samples):
    S_v = librosa.stft(vocals, n_fft=N_FFT, hop_length=HOP_STFT)
    S_i = librosa.stft(accomp, n_fft=N_FFT, hop_length=HOP_STFT)
    T = S_v.shape[1]
    def fit(m): return m[:, :T] if m.shape[1] >= T else np.pad(m, ((0,0),(0,T-m.shape[1])))
    S_v_out = 0.95 * fit(vocal_mask) * S_v + 0.05 * S_v
    S_i_out = 1.00 * fit(inst_mask)  * S_i
    v_wave = librosa.istft(S_v_out, hop_length=HOP_STFT, length=n_samples)
    i_wave = librosa.istft(S_i_out, hop_length=HOP_STFT, length=n_samples)
    return v_wave, i_wave


# ── separation on noisy audio ─────────────────────────────────────────────────

def separate_noisy(audio: np.ndarray, separator, n_sample: int):
    """Run DSP+NN separation and return (vocals, instrs, seg_stats)."""
    pad   = CONTEXT_LEN - (n_sample % HOP_LEN or HOP_LEN)
    audio = np.pad(audio, ((0, 0), (0, pad)))
    n_bins = N_FFT // 2 + 1
    masker = GPUDSPMasker(n_bins, DEVICE)
    wola_v = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    wola_i = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    starts = np.arange(0, audio.shape[1] - CONTEXT_LEN + 1, HOP_LEN)
    chunks_per_hop = HOP_LEN // CHUNK_LEN
    seg_stats = []

    for start in starts:
        seg = audio[:, start:start + CONTEXT_LEN]
        vm, im, hm_arr, pm_arr = dsp_masks(seg, masker)
        vf = float(vm.mean()); if_= float(im.mean())
        hf = float(hm_arr.mean()); pf = float(pm_arr.mean())
        seg_stats.append({
            "vocal_conf": vf / (vf + if_ + 1e-8),
            "perc_conf":  pf / (hf + pf + 1e-8),
            "harm_conf":  hf / (hf + pf + 1e-8),
        })
        tensor = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            estimates = separator(tensor)
        v_raw = estimates[0, 0].mean(0).cpu().numpy()
        i_raw = estimates[0, 1].mean(0).cpu().numpy()
        v_seg, i_seg = apply_dsp_post(v_raw, i_raw, vm, im, CONTEXT_LEN)
        wola_v.push(v_seg)
        wola_i.push(i_seg)

    vocals = wola_v.flush()[:n_sample]
    instrs = wola_i.flush()[:n_sample]

    # Expand seg stats to per-chunk
    n_chunks = int(np.ceil(n_sample / CHUNK_LEN))
    vocal_conf = np.repeat([s["vocal_conf"] for s in seg_stats], chunks_per_hop)[:n_chunks]
    perc_conf  = np.repeat([s["perc_conf"]  for s in seg_stats], chunks_per_hop)[:n_chunks]
    harm_conf  = np.repeat([s["harm_conf"]  for s in seg_stats], chunks_per_hop)[:n_chunks]

    return vocals, instrs, vocal_conf, perc_conf, harm_conf


# ── FEC simulation ─────────────────────────────────────────────────────────────

def is_silent(chunk):
    return float(np.sqrt(np.mean(chunk ** 2))) < SILENCE_THRESH

def all_copies_lost(n_copies, loss_rate, rng):
    return bool((rng.random(n_copies) < loss_rate).all())

def simulate_fec(vocal_chunks, inst_chunks, confidence, silence_mask,
                 perc_conf, harm_conf, loss_rate, rng, mode):
    n = len(vocal_chunks)
    vocal_out = np.zeros_like(vocal_chunks)
    inst_out  = np.zeros_like(inst_chunks)
    v_rec = i_rec = 0
    for idx in range(n):
        if silence_mask[idx]:
            vocal_out[idx] = vocal_chunks[idx]
            inst_out[idx]  = inst_chunks[idx]
            v_rec += 1; i_rec += 1
            continue
        if mode == "eep":
            v_r = i_r = REDUND_BUDGET // 2
        else:
            surplus = REDUND_BUDGET - 2 * MIN_REDUND
            v_score = confidence[idx] + VOCAL_BIAS - harm_conf[idx] * 0.10
            i_score = max(1.0 - confidence[idx], 0.0) + perc_conf[idx] * PERC_WEIGHT
            total   = max(v_score + i_score, 1e-8)
            v_r     = MIN_REDUND + round((v_score / total) * surplus)
            i_r     = REDUND_BUDGET - v_r
        if not all_copies_lost(v_r + 1, loss_rate, rng):
            vocal_out[idx] = vocal_chunks[idx]; v_rec += 1
        if not all_copies_lost(i_r + 1, loss_rate, rng):
            inst_out[idx] = inst_chunks[idx]; i_rec += 1
    return vocal_out, inst_out, v_rec / n, i_rec / n


# ── quality metric ─────────────────────────────────────────────────────────────

def snr_score(ref, deg):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    noise       = ref - deg
    sig_power   = np.mean(ref ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_db      = 10 * np.log10(sig_power / noise_power)
    return round(float(np.clip(snr_db / 40.0 * 100, 0, 100)), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    input_dir   = Path("input")
    clean_dir   = Path("separated_dsp_nn")
    output_base = Path("noise_robustness")
    output_base.mkdir(exist_ok=True)

    mp3s = sorted(input_dir.glob("*.mp3"))
    if not mp3s:
        print("No MP3s found in input/"); return

    print("=" * 72)
    print("  NOISE ROBUSTNESS TEST")
    print("  DSP+NN separation under white Gaussian noise")
    print(f"  Noise levels: {NOISE_LEVELS} dB input SNR")
    print(f"  Loss rates:   {[f'{r*100:.0f}%' for r in LOSS_RATES]}")
    print(f"  Device: {DEVICE}")
    print("=" * 72)

    print(f"\n  Loading Open-Unmix (umxhq) on {DEVICE}...")
    separator = openunmix.umxhq(targets=["vocals"], device=DEVICE, residual=True)
    separator = separator.to(DEVICE)
    separator.eval()

    rng = np.random.default_rng(42)

    for mp3 in mp3s:
        song = mp3.stem
        clean_v_path = clean_dir / song / "streaming_vocals.wav"
        clean_i_path = clean_dir / song / "streaming_instrumentals.wav"

        if not clean_v_path.exists():
            print(f"\n  Skipping {song} — no clean reference in {clean_dir}/")
            continue

        print(f"\n{'='*72}")
        print(f"  {song}")
        print(f"{'='*72}")

        audio    = load_stereo(mp3)
        n_sample = audio.shape[1]
        ref_v    = load_wav(clean_v_path)
        ref_i    = load_wav(clean_i_path)

        print(f"\n  {'Noise':>8}  {'Loss':>6}  {'Mode':<6}  "
              f"{'V.rec':>7}  {'I.rec':>7}  "
              f"{'V.qual':>8}  {'I.qual':>8}  "
              f"{'V.vs.clean':>11}  {'I.vs.clean':>11}")
        print(f"  {'─'*82}")

        for snr_db in NOISE_LEVELS:
            noisy = add_noise(audio, snr_db)

            vocals, instrs, vocal_conf, perc_conf, harm_conf = separate_noisy(
                noisy, separator, n_sample
            )

            vocal_chunks, orig_len = chunk_audio(vocals)
            inst_chunks,  _        = chunk_audio(instrs)
            n_chunks = len(vocal_chunks)
            silence_mask = np.array([is_silent(vocal_chunks[i]) & is_silent(inst_chunks[i])
                                     for i in range(n_chunks)])

            for loss_rate in LOSS_RATES:
                for mode in ["eep", "uep"]:
                    v_out, i_out, v_rec, i_rec = simulate_fec(
                        vocal_chunks, inst_chunks, vocal_conf, silence_mask,
                        perc_conf, harm_conf, loss_rate, rng, mode
                    )
                    v_flat = v_out.flatten()[:orig_len]
                    i_flat = i_out.flatten()[:orig_len]

                    # Quality vs noisy-separated reference (how well FEC recovered)
                    v_q = snr_score(vocals[:orig_len], v_flat)
                    i_q = snr_score(instrs[:orig_len], i_flat)
                    # Quality vs clean reference (end-to-end degradation)
                    v_clean = snr_score(ref_v[:orig_len], v_flat)
                    i_clean = snr_score(ref_i[:orig_len], i_flat)

                    label = f"{snr_db:>2}dB" if mode == "eep" else "    "
                    print(f"  {label:>8}  {loss_rate*100:>5.0f}%  {mode.upper():<6}  "
                          f"{v_rec*100:>6.1f}%  {i_rec*100:>6.1f}%  "
                          f"{v_q:>8.1f}  {i_q:>8.1f}  "
                          f"{v_clean:>11.1f}  {i_clean:>11.1f}")

                    # Save recovered audio
                    out_dir = (output_base / song /
                               f"snr_{snr_db:02d}db" /
                               f"loss_{int(loss_rate*100):02d}pct_{mode.upper()}")
                    os.makedirs(out_dir, exist_ok=True)
                    save_wav(out_dir / "vocals.wav",        v_flat)
                    save_wav(out_dir / "instrumentals.wav", i_flat)

                print()

    print("=" * 72)
    print(f"  Done. Outputs in noise_robustness/")
    print("=" * 72)


if __name__ == "__main__":
    main()
