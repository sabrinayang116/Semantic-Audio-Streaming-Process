"""
Neural-Only Music Source Separator (no DSP)
============================================
Open-Unmix (umxhq) BiLSTM + WOLA 20ms streaming.
No DSP masks — pure neural network output.

Output: separated_nn_only/<song>/vocals.wav
        separated_nn_only/<song>/instrumentals.wav
"""

import os
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import openunmix
import soundfile as sf

SR          = 44100
CHUNK_MS    = 20
CHUNK_LEN   = int(SR * CHUNK_MS / 1000)   # 882 samples
CONTEXT_SEC = 1.0
CONTEXT_LEN = int(SR * CONTEXT_SEC)
OVERLAP     = 0.5
HOP_LEN     = int(CONTEXT_LEN * (1 - OVERLAP))

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


def load_stereo(path: Path):
    y, sr = librosa.load(str(path), sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)   # (2, n_samples)


def save_wav(path: Path, wave: np.ndarray, sr: int = SR):
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    sf.write(str(path), (wave * 32767).astype(np.int16), sr)


class WOLAStream:
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


def load_model():
    print(f"  Loading Open-Unmix (umxhq) on {DEVICE}...")
    separator = openunmix.umxhq(targets=["vocals"], device=DEVICE, residual=True)
    separator = separator.to(DEVICE)
    separator.eval()
    # Confirm model is on the right device
    p = next(separator.parameters())
    print(f"  Model device confirmed: {p.device}")
    return separator


def run_unmix(separator, stereo_np: np.ndarray):
    tensor = torch.from_numpy(stereo_np).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        estimates = separator(tensor)   # (1, 2, 2, n_samples)
    vocals = estimates[0, 0].mean(0).cpu().numpy()
    accomp = estimates[0, 1].mean(0).cpu().numpy()
    return vocals, accomp


def separate(input_path, output_dir, separator):
    song = Path(input_path).stem
    out  = Path(output_dir) / song
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {song}  [NN ONLY — no DSP]")
    print(f"{'='*60}")

    print("  [1/3] Loading audio...")
    audio    = load_stereo(Path(input_path))
    n_sample = audio.shape[1]

    pad   = CONTEXT_LEN - (n_sample % HOP_LEN or HOP_LEN)
    audio = np.pad(audio, ((0, 0), (0, pad)))

    wola_v = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    wola_i = WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)

    starts = np.arange(0, audio.shape[1] - CONTEXT_LEN + 1, HOP_LEN)
    n_segs = len(starts)
    t_nn   = 0.0

    print(f"  [2/3] Separating: {n_segs} × {CONTEXT_SEC*1000:.0f}ms segments → {CHUNK_MS}ms frames...")
    for idx, start in enumerate(starts):
        seg = audio[:, start:start + CONTEXT_LEN]

        t0 = time.perf_counter()
        v_seg, i_seg = run_unmix(separator, seg)
        t_nn += time.perf_counter() - t0

        wola_v.push(v_seg)
        wola_i.push(i_seg)

        if (idx + 1) % max(1, n_segs // 8) == 0 or idx == n_segs - 1:
            per_seg = t_nn / (idx + 1) * 1000
            print(f"    seg {idx+1:4d}/{n_segs}  |  "
                  f"NN {t_nn:.1f}s  |  "
                  f"NN/seg {per_seg:.1f}ms  ({'✓' if per_seg < 20 else '!'} <20ms)")

    print("  [3/3] Saving...")
    vocals = wola_v.flush()[:n_sample]
    instrs = wola_i.flush()[:n_sample]
    save_wav(out / "vocals.wav",        vocals)
    save_wav(out / "instrumentals.wav", instrs)

    print(f"  Done  NN {t_nn:.1f}s | NN/seg {t_nn/n_segs*1000:.1f}ms")
    print(f"  Output: {out}/")


def main():
    input_dir  = Path("input")
    output_dir = Path("separated_nn_only")

    mp3s = sorted(input_dir.glob("*.mp3"))
    if not mp3s:
        print("No MP3 files found in input/")
        return

    print("=" * 60)
    print("  NEURAL-ONLY SOURCE SEPARATOR")
    print("  Open-Unmix (umxhq) · WOLA 20ms · NO DSP")
    print(f"  Device: {DEVICE}")
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
