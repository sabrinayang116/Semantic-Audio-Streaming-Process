"""
FEC Simulation — EEP vs UEP with ViSQOL Quality Metric
=======================================================
Loads separated vocal and instrumental streams from separated_dsp_nn/
Simulates Bernoulli packet loss at 1%, 5%, 10%, 20% loss rates.

EEP (Equal Error Protection):
  Both streams receive the same number of redundant copies — naive baseline.

UEP (Unequal Error Protection):
  Redundancy allocated dynamically per 20ms chunk based on vocal confidence
  score (energy in 300-3000 Hz). Vocals get more protection; instrumentals
  accept higher loss in exchange for better vocal recovery.

FEC model: repetition coding.
  Each packet gets R redundant copies. Recovery if any copy survives.
  P(total loss) = loss_rate ^ (R + 1)

ViSQOL: compares recovered audio vs original clean separation (0-100).
  Requires 48kHz — audio is resampled before scoring.
  Falls back to PESQ if visqol-python is unavailable.

Output: fec_output/<song>/loss_<X>pct_<EEP|UEP>/vocals.wav + instrumentals.wav
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

SR            = 44100
VISQOL_SR     = 48000
CHUNK_MS      = 20
CHUNK_LEN     = int(SR * CHUNK_MS / 1000)
LOSS_RATES    = [0.01, 0.05, 0.10, 0.20]
FEC_OVERHEAD  = 0.25  # redundant copies as a fraction of original packets (25%)
VOCAL_BIAS    = 0.10
PERC_WEIGHT   = 0.30  # extra weight for percussive content → protect instrumentals more
N_FFT         = 2048
HOP_STFT      = 512
SILENCE_THRESH = 1e-4  # RMS below this = silent chunk, skip redundancy
MEAN_BURST    = 4      # Gilbert-Elliott: average burst length in packets (80 ms)

# Staggered placement: copies ride behind the original at these packet offsets.
# UEP gives vocals wider time diversity than instrumentals; EEP is uniform.
EEP_OFFSETS   = [4, 8, 12, 16]
VOCAL_OFFSETS = [6, 12, 18, 24]   # jitter buffer: 24 packets = 480 ms
INST_OFFSETS  = [4, 8, 12, 16]
MAX_COPIES    = 4      # per-stream ceiling (bounds the jitter buffer)
LISTENER_W_V  = 0.6    # listener-weighted score: 0.6×vocal + 0.4×instrumental


# ── ViSQOL setup ──────────────────────────────────────────────────────────────
# ViSQOL runs ~1:1 real-time, so full songs are too slow for a 144-row table.
# We score a 30 s window per measurement and fan out over worker processes.

VISQOL_WINDOW_SEC = 30    # excerpt length to score
VISQOL_START_SEC  = 30    # skip the intro; score an active part of the song
VISQOL_WORKERS    = 6     # parallel scoring processes
MOS_CEILING       = 4.732 # ViSQOL audio-mode maximum (perfect copy)


def mos_to_100(mos: float) -> float:
    return round((mos - 1.0) / (MOS_CEILING - 1.0) * 100, 1)


_worker_api = None

def _visqol_init():
    """Runs once in each worker process: build its ViSQOL API instance."""
    global _worker_api
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = VISQOL_SR
    config.options.use_speech_scoring = False
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__),
        "model", "libsvm_nu_svr_model.txt")
    _worker_api = visqol_lib_py.VisqolApi()
    _worker_api.Create(config)


def _visqol_measure(job):
    """Worker task: (ref_window, deg_window) at 44.1 kHz → score /100."""
    ref, deg = job
    ref48 = librosa.resample(ref, orig_sr=SR, target_sr=VISQOL_SR).astype(np.float64)
    deg48 = librosa.resample(deg, orig_sr=SR, target_sr=VISQOL_SR).astype(np.float64)
    try:
        return mos_to_100(_worker_api.Measure(ref48, deg48).moslqo)
    except Exception:
        return None


def make_visqol_pool():
    try:
        import visqol  # noqa: F401 — confirm the build is importable
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(VISQOL_WORKERS, initializer=_visqol_init)
        print(f"  ViSQOL loaded (audio mode, {VISQOL_WORKERS} workers, "
              f"{VISQOL_WINDOW_SEC}s window)")
        return pool
    except Exception as e:
        print(f"  ViSQOL unavailable ({e}) — falling back to SNR")
        return None


def score_window(audio: np.ndarray) -> np.ndarray:
    dur   = len(audio) / SR
    start = VISQOL_START_SEC if dur > VISQOL_START_SEC + VISQOL_WINDOW_SEC else 0
    s     = int(start * SR)
    return audio[s:s + int(VISQOL_WINDOW_SEC * SR)]


# ── CDPAM setup ───────────────────────────────────────────────────────────────
# CDPAM (Manocha et al. 2021): deep perceptual audio distance trained on human
# judgments. Distance 0 = perceptually identical. We anchor the 0-100 scale per
# stream: 100 = identical to reference, 0 = as far from it as total silence.

CDPAM_SR      = 22050   # CDPAM operates on 22.05 kHz audio
CDPAM_SEG_SEC = 3.0     # score in 3 s segments and average (bounds memory)


def make_cdpam():
    try:
        import torch
        import cdpam
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        scorer = cdpam.CDPAM(dev=dev)
        print(f"  CDPAM loaded ({dev}, {VISQOL_WINDOW_SEC}s window, "
              f"{CDPAM_SEG_SEC:.0f}s segments)")
        return scorer
    except Exception as e:
        print(f"  CDPAM unavailable ({e}) — falling back to SNR")
        return None


def _cdpam_prep(wav44: np.ndarray):
    import torch
    wav = librosa.resample(wav44.astype(np.float32), orig_sr=SR, target_sr=CDPAM_SR)
    return torch.tensor(np.round(wav * 32768.0), dtype=torch.float32).unsqueeze(0)


def cdpam_distance(scorer, ref44: np.ndarray, deg44: np.ndarray) -> float:
    """Mean CDPAM distance across CDPAM_SEG_SEC segments of the window."""
    import torch
    seg   = int(CDPAM_SEG_SEC * SR)
    dists = []
    for s in range(0, len(ref44) - seg + 1, seg):
        r = _cdpam_prep(ref44[s:s + seg])
        d = _cdpam_prep(deg44[s:s + seg])
        with torch.no_grad():
            dists.append(float(scorer.forward(r, d)))
    return float(np.mean(dists)) if dists else 0.0


def cdpam_score(dist: float, anchor: float) -> float:
    """Map distance to 0-100: 100 = identical, 0 = the total-loss anchor."""
    return round(float(np.clip(1.0 - dist / (anchor + 1e-12), 0.0, 1.0)) * 100, 1)


# ── SI-SDR (per-stream) ───────────────────────────────────────────────────────
# Scale-invariant signal-to-distortion ratio — the source-separation standard.
# Scaled 0-100: 0 dB → 0, 30 dB → 100 (near-perfect recovery caps at 100).

def si_sdr_100(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    alpha  = np.dot(deg, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise  = deg - target
    val = 10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))
    return round(float(np.clip(val, 0, 30) / 30.0 * 100), 1)


# ── PEAQ (combined mix) ───────────────────────────────────────────────────────
# ITU-R BS.1387 psychoacoustic ear model, built from gstpeaq source.
# Output ODG: 0 = imperceptible difference, -4 = very annoying → 0-100.

PEAQ_BIN = "/Users/sabrinayang/semantic_env/tools/peaq/peaq"
PEAQ_ENV = {
    **os.environ,
    "GST_PLUGIN_PATH":  "/Users/sabrinayang/semantic_env/tools/peaq/.libs",
    "DYLD_LIBRARY_PATH": "/opt/homebrew/opt/gstreamer/lib:" + os.environ.get("DYLD_LIBRARY_PATH", ""),
}


def peaq_available() -> bool:
    return os.path.exists(PEAQ_BIN)


def peaq_100(ref: np.ndarray, deg: np.ndarray) -> float:
    """Write the mix pair to temp WAVs, run PEAQ, map ODG → 0-100."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        rp, dp = os.path.join(td, "r.wav"), os.path.join(td, "d.wav")
        for path, sig in ((rp, ref), (dp, deg)):
            peak = np.max(np.abs(sig)) or 1.0
            sf.write(path, (sig / peak * 0.9 * 32767).astype(np.int16), SR)
        try:
            out = subprocess.run([PEAQ_BIN, "--basic", rp, dp],
                                 env=PEAQ_ENV, capture_output=True, text=True, timeout=120)
            for line in out.stdout.splitlines():
                if "Objective Difference Grade" in line:
                    odg = float(line.split(":")[1])
                    return round(float(np.clip((odg + 4.0) / 4.0 * 100, 0, 100)), 1)
        except Exception:
            pass
    return None


# ── Meta Audiobox Aesthetics (combined mix, reference-free) ────────────────────
# Neural aesthetic predictor trained on human MOS. Uses the Production Quality
# (PQ) axis, roughly 1-10 → 0-100. Reference-free: rates the degraded mix alone.

def make_audiobox():
    try:
        from audiobox_aesthetics.infer import initialize_predictor
        p = initialize_predictor()
        print("  Audiobox Aesthetics loaded (Production Quality axis)")
        return p
    except Exception as e:
        print(f"  Audiobox unavailable ({e})")
        return None


def audiobox_100(predictor, mix: np.ndarray) -> float:
    import torch
    try:
        wav = torch.tensor(mix.astype(np.float32)).unsqueeze(0)
        out = predictor.forward([{"path": wav, "sample_rate": SR}])
        pq  = out[0]["PQ"]                       # production quality, ~1-10
        return round(float(np.clip(pq / 10.0 * 100, 0, 100)), 1)
    except Exception:
        return None


def compute_snr_score(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Pure numpy SNR scaled to 0-100. No C extensions — cannot crash kernel.
    SNR = 10*log10(signal power / noise power), clamped to [0, 40] dB → [0, 100].
    Perfect recovery (no loss) = 100. Complete loss (zeros) ≈ 0.
    """
    n   = min(len(ref), len(deg))
    ref = ref[:n]
    deg = deg[:n]
    noise      = ref - deg
    sig_power  = np.mean(ref ** 2) + 1e-10
    noise_power= np.mean(noise ** 2) + 1e-10
    snr_db     = 10 * np.log10(sig_power / noise_power)
    return round(float(np.clip(snr_db / 40.0 * 100, 0, 100)), 1)


def fmt_score(score) -> str:
    return f"{score:>5.1f}" if score is not None else "  N/A"


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    sf.write(str(path), (audio * 32767).astype(np.int16), sr)


def chunk_audio(audio: np.ndarray, chunk_len: int):
    n        = len(audio)
    n_chunks = int(np.ceil(n / chunk_len))
    padded   = np.zeros(n_chunks * chunk_len, dtype=np.float32)
    padded[:n] = audio
    return padded.reshape(n_chunks, chunk_len), n


# ── per-chunk vocal confidence ─────────────────────────────────────────────────

def is_silent(chunk: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(chunk ** 2))) < SILENCE_THRESH


def vocal_confidence_score(chunk: np.ndarray) -> float:
    S      = np.abs(librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_STFT))
    freqs  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    v_bins = (freqs >= 300) & (freqs <= 3000)
    return float(S[v_bins].sum() / (S.sum() + 1e-8))


# ── FEC simulation ─────────────────────────────────────────────────────────────

def all_copies_lost(n_copies: int, loss_rate: float, rng) -> bool:
    return bool((rng.random(n_copies) < loss_rate).all())


def gen_gilbert_loss(n_slots: int, loss_rate: float, rng) -> np.ndarray:
    """
    Gilbert-Elliott burst channel: packets are lost in consecutive runs.
    Two states — good (no loss) and bad (loss). Mean burst length is
    MEAN_BURST packets; stationary loss probability equals loss_rate.
    """
    p_bg = 1.0 / MEAN_BURST                        # P(bad → good)
    p_gb = p_bg * loss_rate / (1.0 - loss_rate)    # P(good → bad)
    loss = np.zeros(n_slots, dtype=bool)
    bad  = rng.random() < loss_rate
    for t in range(n_slots):
        loss[t] = bad
        bad = (rng.random() >= p_bg) if bad else (rng.random() < p_gb)
    return loss


def simulate_fec(vocal_chunks, inst_chunks, confidence, silence_mask,
                 perc_conf, harm_conf, loss_rate, rng, mode,
                 channel="bernoulli", placement="staggered"):
    n         = len(vocal_chunks)
    vocal_out = np.zeros_like(vocal_chunks)
    inst_out  = np.zeros_like(inst_chunks)
    v_rec = i_rec = 0

    slot_loss = None
    max_off   = max(VOCAL_OFFSETS[-1], INST_OFFSETS[-1], EEP_OFFSETS[-1])
    if channel == "burst":
        slot_loss = gen_gilbert_loss(n + max_off + 1, loss_rate, rng)

    # Combined chunk loudness → "quiet passage" detection for UEP banking
    rms = np.sqrt(np.mean(vocal_chunks ** 2, axis=1) +
                  np.mean(inst_chunks ** 2, axis=1))
    active = ~silence_mask
    active_median = float(np.median(rms[active])) if active.any() else 0.0

    def survives(idx, n_extra, offsets):
        if channel == "bernoulli":
            # every copy is an independent packet — placement is irrelevant
            return not all_copies_lost(n_extra + 1, loss_rate, rng)
        if placement == "continuous":
            # original + copies sent back-to-back → one burst takes them all
            return not slot_loss[idx]
        # staggered: copies ride in later packet slots (time diversity), so a
        # burst that kills the original is unlikely to also cover every copy
        slots = [idx] + [idx + off for off in offsets[:n_extra]]
        return not all(slot_loss[s] for s in slots)

    # ── Allocation pass ───────────────────────────────────────────────────
    # 25% FEC overhead: the pool is FEC_OVERHEAD × (2n originals) redundant
    # copies for the whole song — 0.5 copies per chunk on average, so most
    # chunks get no protection at all. Both modes spend exactly this pool.
    v_alloc = np.zeros(n, dtype=int)
    i_alloc = np.zeros(n, dtype=int)
    pool    = int(round(FEC_OVERHEAD * 2 * n))

    # Per-chunk stream scores (used by UEP; v_fracs also picks the copy target)
    v_sc = confidence + VOCAL_BIAS - harm_conf * 0.10
    i_sc = np.maximum(1.0 - confidence, 0.0) + perc_conf * PERC_WEIGHT
    v_fracs = v_sc / np.maximum(v_sc + i_sc, 1e-8)

    if mode == "eep":
        # EEP: content-blind uniform spread — one copy every other chunk,
        # alternating streams, until the pool is spent
        spent = 0
        for idx in range(n):
            if spent >= pool:
                break
            if idx % 4 == 0:
                v_alloc[idx] = 1
                spent += 1
            elif idx % 4 == 2:
                i_alloc[idx] = 1
                spent += 1
    else:
        # UEP: triage — rank every active chunk by the importance of its
        # dominant stream, protect the top `pool` chunks, skip silence and
        # deprioritise quiet passages. Same pool as EEP, spent by content.
        chunk_imp = np.maximum(v_sc, i_sc)
        chunk_imp = np.where(active, chunk_imp, -1.0)          # never protect silence
        quiet_mask = active & (rms < 0.3 * active_median)
        chunk_imp = np.where(quiet_mask, chunk_imp * 0.5, chunk_imp)

        order = np.argsort(-chunk_imp)
        spent = 0
        # First pass: one copy to the dominant stream of the top chunks
        for idx in order:
            if spent >= pool or chunk_imp[idx] < 0:
                break
            if v_fracs[idx] >= 0.5:
                v_alloc[idx] = 1
            else:
                i_alloc[idx] = 1
            spent += 1
        # If pool exceeds active chunks, second copy to the other stream
        if spent < pool:
            for idx in order:
                if spent >= pool or chunk_imp[idx] < 0:
                    break
                if v_alloc[idx] == 0:
                    v_alloc[idx] = 1
                    spent += 1
                elif i_alloc[idx] == 0:
                    i_alloc[idx] = 1
                    spent += 1

    # ── Transmission pass ─────────────────────────────────────────────────
    for idx in range(n):
        if mode != "eep" and silence_mask[idx]:
            # silent chunks pass through — losing them is perceptually free
            vocal_out[idx] = vocal_chunks[idx]
            inst_out[idx]  = inst_chunks[idx]
            v_rec += 1
            i_rec += 1
            continue

        v_off = EEP_OFFSETS if mode == "eep" else VOCAL_OFFSETS
        i_off = EEP_OFFSETS if mode == "eep" else INST_OFFSETS

        if survives(idx, v_alloc[idx], v_off):
            vocal_out[idx] = vocal_chunks[idx]
            v_rec += 1

        if survives(idx, i_alloc[idx], i_off):
            inst_out[idx] = inst_chunks[idx]
            i_rec += 1

    return vocal_out, inst_out, v_rec / n, i_rec / n


# ── per-song runner ────────────────────────────────────────────────────────────

def run_song(song_dir: Path, output_base: Path, visqol_pool, cdpam, audiobox):
    vocal_path = song_dir / "streaming_vocals.wav"
    inst_path  = song_dir / "streaming_instrumentals.wav"

    if not vocal_path.exists() or not inst_path.exists():
        print(f"  Skipping {song_dir.name} — missing separated files")
        return

    print(f"\n{'='*72}")
    print(f"  {song_dir.name}")
    print(f"{'='*72}")

    vocals = load_wav(vocal_path)
    instrs = load_wav(inst_path)
    n      = min(len(vocals), len(instrs))
    vocals, instrs = vocals[:n], instrs[:n]

    vocal_chunks, orig_len = chunk_audio(vocals, CHUNK_LEN)
    inst_chunks,  _        = chunk_audio(instrs, CHUNK_LEN)
    n_chunks = len(vocal_chunks)
    print(f"  Packets: {n_chunks} × {CHUNK_MS}ms")

    silence_mask = np.array([is_silent(vocal_chunks[i]) & is_silent(inst_chunks[i])
                             for i in range(n_chunks)])
    n_silent = silence_mask.sum()

    print("  Computing per-chunk vocal confidence scores...")
    confidence = np.array([vocal_confidence_score(c) for c in vocal_chunks])

    dsp_path = song_dir / "dsp_masks.npz"
    if dsp_path.exists():
        dsp = np.load(dsp_path)
        perc_conf = dsp["perc_conf"][:n_chunks]
        harm_conf = dsp["harm_conf"][:n_chunks]
        print(f"  DSP mask stats loaded from dsp_masks.npz")
    else:
        print("  dsp_masks.npz not found — no percussive/harmonic weighting")
        perc_conf = np.zeros(n_chunks, dtype=np.float32)
        harm_conf = np.zeros(n_chunks, dtype=np.float32)

    print(f"  Mean vocal conf: {confidence.mean():.3f}  |  "
          f"Mean perc: {perc_conf.mean():.3f}  |  "
          f"Mean harm: {harm_conf.mean():.3f}  |  "
          f"Silent: {n_silent}/{n_chunks} ({100*n_silent/n_chunks:.1f}%)")

    rng = np.random.default_rng(42)
    combos = [("bernoulli", "staggered"), ("burst", "continuous"), ("burst", "staggered")]

    # Reference windows: per-stream (for SI-SDR) and combined mix (for the rest)
    ref_v_win   = score_window(vocals)
    ref_i_win   = score_window(instrs)
    ref_mix_win = score_window(vocals + instrs)

    # Pass 1 — simulate, save wavs (incl. recombined mix), compute SI-SDR inline,
    # collect combined-mix windows for the reference-based mix metrics
    rows, visqol_jobs, deg_mix_wins = [], [], []

    for loss_rate in LOSS_RATES:
        for channel, placement in combos:
            for mode in ["eep", "uep"]:
                v_out, i_out, v_rec, i_rec = simulate_fec(
                    vocal_chunks, inst_chunks, confidence, silence_mask,
                    perc_conf, harm_conf, loss_rate, rng, mode,
                    channel=channel, placement=placement
                )
                v_flat  = v_out.flatten()[:orig_len]
                i_flat  = i_out.flatten()[:orig_len]
                mix_deg = v_flat + i_flat                      # recombined song

                # SI-SDR per stream (instant, on the 30 s window)
                sdr_v = si_sdr_100(ref_v_win, score_window(v_flat))
                sdr_i = si_sdr_100(ref_i_win, score_window(i_flat))

                deg_mix_win = score_window(mix_deg)
                visqol_jobs.append((ref_mix_win, deg_mix_win))
                deg_mix_wins.append(deg_mix_win)

                place_lbl = "—" if channel == "bernoulli" else placement
                # [loss,chan,place,mode,vrec,irec, sdrV,sdrI, visqol,cdpam,peaq,abox]
                rows.append([loss_rate, channel, place_lbl, mode, v_rec, i_rec,
                             sdr_v, sdr_i, None, None, None, None])

                out_dir = (output_base / song_dir.name /
                           f"loss_{int(loss_rate*100):02d}pct_{channel}_{place_lbl}_{mode.upper()}")
                os.makedirs(out_dir, exist_ok=True)
                write_wav(out_dir / "vocals.wav",        v_flat)
                write_wav(out_dir / "instrumentals.wav", i_flat)
                write_wav(out_dir / "mix.wav",           mix_deg)

    # Pass 2 — combined-mix metrics on the reconstructed song
    n = len(rows)

    if visqol_pool is not None:
        print(f"  ViSQOL: {n} measurements ({VISQOL_WORKERS} workers)...")
        vq = visqol_pool.map(_visqol_measure, visqol_jobs)
        for i, row in enumerate(rows):
            row[8] = vq[i]

    if cdpam is not None:
        print(f"  CDPAM: {n} measurements...")
        anchor = cdpam_distance(cdpam, ref_mix_win, np.zeros_like(ref_mix_win))
        for i, deg in enumerate(deg_mix_wins):
            rows[i][9] = cdpam_score(cdpam_distance(cdpam, ref_mix_win, deg), anchor)

    if peaq_available():
        print(f"  PEAQ: {n} measurements...")
        for i, deg in enumerate(deg_mix_wins):
            rows[i][10] = peaq_100(ref_mix_win, deg)

    if audiobox is not None:
        print(f"  Audiobox: {n} measurements...")
        for i, deg in enumerate(deg_mix_wins):
            rows[i][11] = audiobox_100(audiobox, deg)

    hdr = (f"\n  {'Loss':>5}  {'Channel':<10}  {'Placement':<11}  {'Mode':<5}  "
           f"{'SDR-V':>6}  {'SDR-I':>6}  {'ViSQOL':>7}  {'CDPAM':>6}  {'PEAQ':>6}  {'ABox':>6}")
    print(hdr)
    print(f"  {'─'*94}")

    prev_loss = None
    for r in rows:
        loss_rate, channel, place_lbl, mode, v_rec, i_rec, sdr_v, sdr_i, vq, cd, pq, ab = r
        if prev_loss is not None and loss_rate != prev_loss:
            print()
        prev_loss = loss_rate
        print(f"  {loss_rate*100:>4.0f}%  {channel:<10}  {place_lbl:<11}  {mode.upper():<5}  "
              f"{fmt_score(sdr_v):>6}  {fmt_score(sdr_i):>6}  {fmt_score(vq):>7}  "
              f"{fmt_score(cd):>6}  {fmt_score(pq):>6}  {fmt_score(ab):>6}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    input_base  = Path("separated_dsp_nn")
    output_base = Path("fec_output")

    if not input_base.exists():
        print(f"'{input_base}/' not found.")
        print("Run notebook_dsp_nn.ipynb first and unzip separated_dsp_nn.zip here.")
        return

    songs = sorted(d for d in input_base.iterdir() if d.is_dir())
    if not songs:
        print(f"No song folders found in {input_base}/")
        return

    print("=" * 72)
    print("  FEC SIMULATION — EEP vs UEP, multi-metric evaluation (all scores /100)")
    print(f"  Channels:         Bernoulli (independent) + Gilbert-Elliott bursts (mean {MEAN_BURST} pkts)")
    print(f"  Placement:        continuous (back-to-back) vs staggered (time diversity)")
    print(f"  FEC overhead:     {FEC_OVERHEAD*100:.0f}% ({FEC_OVERHEAD*2:.1f} copies/chunk avg) — both modes spend the identical pool")
    print(f"  Metrics:          SI-SDR per stream (SDR-V, SDR-I);")
    print(f"                    ViSQOL, CDPAM, PEAQ, Audiobox on the recombined song")
    print("=" * 72)

    visqol_pool = make_visqol_pool()
    cdpam       = make_cdpam()
    audiobox    = make_audiobox()
    print(f"  PEAQ (gstpeaq ODG): {'available' if peaq_available() else 'NOT FOUND'}")

    for song_dir in songs:
        run_song(song_dir, output_base, visqol_pool, cdpam, audiobox)

    if visqol_pool is not None:
        visqol_pool.close()
        visqol_pool.join()

    print("=" * 72)
    print(f"  All done. Recovered audio (incl. mix.wav) in fec_output/")
    print("=" * 72)


if __name__ == "__main__":
    main()
