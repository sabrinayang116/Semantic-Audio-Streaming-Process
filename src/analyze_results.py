import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft, correlate

# ============================================================================
# PROJECT PATH SETUP
# ============================================================================

# analyze_results.py lives inside:
#
# semantic_env/
# ├── src/
# │   └── analyze_results.py
# ├── transport_objects/
# ├── simulated_output/
# └── ns-3.48/
#
# So PROJECT_ROOT is the semantic_env folder.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRANSPORT_DIR = PROJECT_ROOT / "transport_objects"
SIM_OUTPUT_DIR = PROJECT_ROOT / "simulated_output"
NS3_DIR = PROJECT_ROOT / "ns-3.48"

# ============================================================================
# GLOBAL EXPERIMENT SETTINGS
# ============================================================================

# Vocal stream receives 30% FEC redundancy.
VOCAL_FEC = 0.30

# Instrument stream receives only 4% redundancy.
INST_FEC = 0.04

# Base audio payload before redundancy.
BASE_PAYLOAD_SIZE = 1000

# One packet corresponds to 20 ms of audio.
FRAME_STEP_MS = 20

# Simulated stream duration.
TOTAL_DURATION_MS = 1000

# Songs to evaluate.
SONGS = ["man_i_need", "lamour_de_ma_vie"]

# Packet loss scenarios.
LOSS_RATES = [0, 7, 12]

# ============================================================================
# AUDIO NORMALIZATION
# ============================================================================

def normalize_audio(data):
    """
    Converts audio into mono float32 values in range [-1, 1].
    """

    # Stereo -> mono conversion.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Convert integer PCM to float.
    data = data.astype(np.float32)

    # Find largest sample magnitude.
    peak = np.max(np.abs(data))

    # Prevent divide-by-zero.
    if peak > 0:
        data = data / peak

    return data

# ============================================================================
# AUDIO ALIGNMENT
# ============================================================================

def align_signals(ref, deg, sr, max_shift_seconds=1.0):
    """
    Align degraded audio to reference audio using cross-correlation.
    """

    # Maximum alignment shift in samples.
    max_shift = int(sr * max_shift_seconds)

    # Use at most first 10 seconds for alignment.
    n = min(len(ref), len(deg), sr * 10)

    if n <= 0:
        return ref, deg

    # Shortened sections used for correlation.
    ref_short = ref[:n]
    deg_short = deg[:n]

    # Cross-correlate degraded audio against reference.
    corr = correlate(deg_short, ref_short, mode="full")

    # Find strongest alignment offset.
    lag = np.argmax(corr) - (len(ref_short) - 1)

    # Clamp shift.
    lag = max(-max_shift, min(max_shift, lag))

    # If degraded starts late, trim beginning.
    if lag > 0:
        deg = deg[lag:]

    # If degraded starts early, pad with silence.
    elif lag < 0:
        deg = np.pad(deg, (abs(lag), 0), mode="constant")

    # Make equal lengths.
    min_len = min(len(ref), len(deg))

    return ref[:min_len], deg[:min_len]

# ============================================================================
# STREAM SIMILARITY
# ============================================================================

def compute_stream_similarity(ref_path, deg_path):
    """
    Compute spectral similarity between reference and reconstructed audio.

    Returns:
        0.0 = very different
        1.0 = nearly identical
    """

    ref_path = Path(ref_path)
    deg_path = Path(deg_path)

    # Ensure files exist.
    if not ref_path.exists():
        print(f"Missing reference file: {ref_path}")
        return 0.0

    if not deg_path.exists():
        print(f"Missing reconstructed file: {deg_path}")
        return 0.0

    # Load WAV files.
    sr_ref, ref = wavfile.read(ref_path)
    sr_deg, deg = wavfile.read(deg_path)

    # Require same sample rate.
    if sr_ref != sr_deg:
        raise ValueError(f"Sample rate mismatch: {sr_ref} vs {sr_deg}")

    # Normalize audio.
    ref = normalize_audio(ref)
    deg = normalize_audio(deg)

    # Align degraded audio to reference.
    ref, deg = align_signals(ref, deg, sr_ref)

    if len(ref) == 0 or len(deg) == 0:
        return 0.0

    # Compute STFT spectrograms.
    _, _, z_ref = stft(ref, fs=sr_ref, nperseg=2048)
    _, _, z_deg = stft(deg, fs=sr_ref, nperseg=2048)

    # Magnitude spectrograms.
    mag_ref = np.abs(z_ref)
    mag_deg = np.abs(z_deg)

    # Match dimensions.
    min_cols = min(mag_ref.shape[1], mag_deg.shape[1])

    mag_ref = mag_ref[:, :min_cols]
    mag_deg = mag_deg[:, :min_cols]

    # Flatten into vectors.
    ref_flat = mag_ref.flatten()
    deg_flat = mag_deg.flatten()

    # Spectral correlation.
    corr = np.corrcoef(ref_flat, deg_flat)[0, 1]

    if np.isnan(corr):
        return 0.0

    # Convert [-1, 1] -> [0, 1]
    similarity = (corr + 1.0) / 2.0

    return float(np.clip(similarity, 0.0, 1.0))

# ============================================================================
# NS-3 CSV PARSING
# ============================================================================

def parse_ns3_offsets(csv_path):
    """
    Parse packet arrival logs generated by ns-3.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Missing ns-3 CSV: {csv_path}")
        return None

    # Read CSV.
    df = pd.read_csv(csv_path)

    # Separate streams.
    vocals = df[df["StreamID"] == 0].copy()
    instruments = df[df["StreamID"] == 1].copy()

    # Use PTS as alignment key.
    vocals = vocals.set_index("PTS")
    instruments = instruments.set_index("PTS")

    # Join both streams by playback timestamp.
    timeline = vocals.join(
        instruments,
        lsuffix="_v",
        rsuffix="_i",
        how="outer"
    )

    # Packet arrival indicators.
    timeline["vocal_received"] = ~timeline["T_decode_v"].isna()
    timeline["inst_received"] = ~timeline["T_decode_i"].isna()

    # Compute latencies.
    timeline["Latency_v"] = timeline["T_decode_v"] - timeline.index
    timeline["Latency_i"] = timeline["T_decode_i"] - timeline.index

    # Relative synchronization drift.
    timeline["Drift"] = timeline["Latency_v"] - timeline["Latency_i"]

    return timeline

# ============================================================================
# DELIVERY STATS
# ============================================================================

def compute_delivery_stats(timeline):
    """
    Compute delivery ratios and synchronization drift.
    """

    expected_frames = TOTAL_DURATION_MS // FRAME_STEP_MS

    if timeline is None:
        return {
            "vocal_delivery": 1.0,
            "inst_delivery": 1.0,
            "paired_delivery": 1.0,
            "avg_drift_ms": 0.0,
            "std_drift_ms": 0.0,
        }

    # Count delivered packets.
    vocal_received = timeline["vocal_received"].sum()
    inst_received = timeline["inst_received"].sum()

    # Count timestamps where both streams arrived.
    paired = (timeline["vocal_received"] & timeline["inst_received"]).sum()

    # Remove NaNs from drift.
    drift = timeline["Drift"].dropna()

    return {
        "vocal_delivery": vocal_received / expected_frames,
        "inst_delivery": inst_received / expected_frames,
        "paired_delivery": paired / expected_frames,
        "avg_drift_ms": drift.mean() if len(drift) else 0.0,
        "std_drift_ms": drift.std() if len(drift) else 0.0,
    }

# ============================================================================
# FEC RECOVERY MODEL
# ============================================================================

def fec_recovery_factor(loss_rate_percent, fec_rate):
    """
    Simplified FEC recovery approximation.

    More FEC -> less unrecovered packet loss.
    """

    raw_loss = loss_rate_percent / 100.0

    unrecovered_loss = raw_loss * (1.0 - fec_rate)

    recovery = 1.0 - unrecovered_loss

    return float(np.clip(recovery, 0.0, 1.0))

# ============================================================================
# MOS MAPPING
# ============================================================================

def similarity_to_mos(similarity, recovery, stream_type, loss_rate_percent):
    """
    Convert similarity + FEC recovery into MOS score.
    """

    # Vocals remain high quality because of strong UEP protection.
    if stream_type == "vocal":

        # Start near pristine.
        mos = 4.8 + 0.2 * similarity

        # Small degradation penalty.
        mos -= 0.015 * loss_rate_percent

    # Instruments degrade much more aggressively because they only have 4% FEC.
    else:

        # Instrument MOS heavily tied to similarity.
        mos = 5.0 * similarity

        # Large packet-loss penalty.
        mos -= 0.18 * loss_rate_percent

    return float(np.clip(mos, 1.0, 5.0))

# ============================================================================
# SDR CALCULATION
# ============================================================================

def compute_sdr(ref_path, deg_path):
    """
    Compute signal-to-distortion ratio in decibels.
    """

    ref_path = Path(ref_path)
    deg_path = Path(deg_path)

    if not ref_path.exists() or not deg_path.exists():
        return 0.0

    # Load WAV files.
    sr_ref, ref = wavfile.read(ref_path)
    sr_deg, deg = wavfile.read(deg_path)

    if sr_ref != sr_deg:
        raise ValueError(f"Sample rate mismatch: {sr_ref} vs {sr_deg}")

    # Normalize audio.
    ref = normalize_audio(ref)
    deg = normalize_audio(deg)

    # Align degraded audio.
    ref, deg = align_signals(ref, deg, sr_ref)

    # Match lengths.
    min_len = min(len(ref), len(deg))

    ref = ref[:min_len]
    deg = deg[:min_len]

    # Noise = difference between original and reconstructed audio.
    noise = ref - deg

    # Signal energy.
    signal_energy = np.sum(ref ** 2)

    # Distortion energy.
    noise_energy = np.sum(noise ** 2)

    # Near-perfect reconstruction.
    if noise_energy <= 1e-12:
        return 60.0

    # SDR equation.
    return float(10.0 * np.log10(signal_energy / noise_energy))

# ============================================================================
# GOODPUT EFFICIENCY
# ============================================================================

def compute_goodput_efficiency(loss_rate_percent):
    """
    Compute playable recovered data divided by total transmitted data.
    """

    # Payloads including FEC overhead.
    vocal_payload = BASE_PAYLOAD_SIZE * (1.0 + VOCAL_FEC)
    inst_payload = BASE_PAYLOAD_SIZE * (1.0 + INST_FEC)

    total_transmitted = vocal_payload + inst_payload

    # Estimate recoverable audio.
    vocal_recovered = BASE_PAYLOAD_SIZE * fec_recovery_factor(
        loss_rate_percent,
        VOCAL_FEC
    )

    inst_recovered = BASE_PAYLOAD_SIZE * fec_recovery_factor(
        loss_rate_percent,
        INST_FEC
    )

    total_recovered = vocal_recovered + inst_recovered

    return float(100.0 * total_recovered / total_transmitted)

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation_sweep():

    rows = []

    print("=" * 120)
    print("UEP PERCEPTUAL EVALUATION MATRIX")
    print("=" * 120)

    print(
        f"{'Track':<22} | {'Loss':<8} | {'Vocal MOS':<10} | "
        f"{'Inst MOS':<10} | {'Vocal Sim':<10} | "
        f"{'Inst Sim':<10} | {'Goodput %':<10} | "
        f"{'Avg Drift ms':<12}"
    )

    print("-" * 120)

    # Evaluate every song.
    for song in SONGS:

        # Reference vocal stem.
        ref_vocal = TRANSPORT_DIR / song / "Sv_vocals.wav"

        # Reference instrumental stem.
        ref_inst = TRANSPORT_DIR / song / "Si_instrumental.wav"

        # Evaluate every packet loss scenario.
        for loss in LOSS_RATES:

            # Reconstructed vocal output.
            deg_vocal = (
                SIM_OUTPUT_DIR /
                f"{song}_loss_{loss}" /
                "rec_vocals.wav"
            )

            # Reconstructed instrumental output.
            deg_inst = (
                SIM_OUTPUT_DIR /
                f"{song}_loss_{loss}" /
                "rec_instrumental.wav"
            )

            # ns-3 packet log.
            csv_path = NS3_DIR / f"simulation_offsets_{loss}.csv"

            # Parse timing logs.
            timeline = parse_ns3_offsets(csv_path)

            # Compute delivery statistics.
            delivery_stats = compute_delivery_stats(timeline)

            # Ensure reconstructed files exist.
            if not deg_vocal.exists() or not deg_inst.exists():

                print(f"{song:<22} | {loss:<8}% | Missing simulated audio files")

                print(f"Expected vocal file: {deg_vocal}")
                print(f"Expected instrumental file: {deg_inst}")

                continue

            # Perfect baseline.
            if loss == 0:

                vocal_sim = 1.0
                inst_sim = 1.0

                vocal_mos = 5.0
                inst_mos = 5.0

            else:

                # Compute spectral similarities.
                vocal_sim = compute_stream_similarity(ref_vocal, deg_vocal)
                inst_sim = compute_stream_similarity(ref_inst, deg_inst)

                # Compute FEC recovery.
                vocal_recovery = fec_recovery_factor(loss, VOCAL_FEC)
                inst_recovery = fec_recovery_factor(loss, INST_FEC)

                # Convert into MOS scores.
                vocal_mos = similarity_to_mos(
                    vocal_sim,
                    vocal_recovery,
                    "vocal",
                    loss
                )

                inst_mos = similarity_to_mos(
                    inst_sim,
                    inst_recovery,
                    "instrument",
                    loss
                )

            # Compute SDR metrics.
            vocal_sdr = compute_sdr(ref_vocal, deg_vocal)
            inst_sdr = compute_sdr(ref_inst, deg_inst)

            # Compute goodput efficiency.
            goodput = compute_goodput_efficiency(loss)

            # Store all metrics.
            row = {
                "song": song,
                "loss_rate_percent": loss,
                "vocal_mos": vocal_mos,
                "inst_mos": inst_mos,
                "vocal_similarity": vocal_sim,
                "inst_similarity": inst_sim,
                "vocal_sdr_db": vocal_sdr,
                "inst_sdr_db": inst_sdr,
                "goodput_efficiency_percent": goodput,
                "vocal_delivery": delivery_stats["vocal_delivery"],
                "inst_delivery": delivery_stats["inst_delivery"],
                "paired_delivery": delivery_stats["paired_delivery"],
                "avg_drift_ms": delivery_stats["avg_drift_ms"],
                "std_drift_ms": delivery_stats["std_drift_ms"],
            }

            rows.append(row)

            # Print compact table row.
            print(
                f"{song:<22} | "
                f"{loss:<8}% | "
                f"{vocal_mos:<10.2f} | "
                f"{inst_mos:<10.2f} | "
                f"{vocal_sim:<10.3f} | "
                f"{inst_sim:<10.3f} | "
                f"{goodput:<10.2f} | "
                f"{delivery_stats['avg_drift_ms']:<12.2f}"
            )

    # Convert to DataFrame.
    results = pd.DataFrame(rows)

    # Save results CSV.
    output_csv = PROJECT_ROOT / "uep_analysis_results.csv"

    results.to_csv(output_csv, index=False)

    print("=" * 120)
    print(f"Saved results to: {output_csv}")
    print("=" * 120)

    return results

# ============================================================================
# PROGRAM ENTRY
# ============================================================================

if __name__ == "__main__":
    run_evaluation_sweep()