"""
Noise Sensitivity Test
======================
How robust is the pipeline (separation + UEP/EEP FEC) to input noise?

For each song: separate the CLEAN input (reference), then add white Gaussian
noise at several input-SNR levels, separate each noisy version, run FEC at a
representative loss condition, and score the recovered song against the clean
reference. Shows how output quality degrades with noise, and whether UEP's
advantage over EEP survives.

Separation ships raw NN audio (metadata-only DSP), matching the main pipeline.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import openunmix
from pathlib import Path

import noise_robustness as N          # add_noise, model, DSP masker, WOLA, dsp_masks
import fec_simulation as F            # metrics + current triage simulate_fec

SR, CHUNK_LEN = N.SR, N.CHUNK_LEN
CONTEXT_LEN, HOP_LEN = N.CONTEXT_LEN, N.HOP_LEN

NOISE_LEVELS = [None, 20, 10, 0]      # None = clean; else input SNR in dB
LOSS_RATE    = 0.10                    # representative loss
CHANNEL, PLACEMENT = "burst", "staggered"


def separate_raw(audio, separator, n_sample):
    """Separate, shipping RAW NN audio; return stems + per-chunk DSP scores."""
    pad   = CONTEXT_LEN - (n_sample % HOP_LEN or HOP_LEN)
    audio = np.pad(audio, ((0, 0), (0, pad)))
    masker = N.GPUDSPMasker(N.N_FFT // 2 + 1, N.DEVICE)
    wola_v = N.WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    wola_i = N.WOLAStream(CONTEXT_LEN, HOP_LEN, CHUNK_LEN)
    starts = np.arange(0, audio.shape[1] - CONTEXT_LEN + 1, HOP_LEN)
    cph, stats = HOP_LEN // CHUNK_LEN, []
    for start in starts:
        seg = audio[:, start:start + CONTEXT_LEN]
        _, _, hm, pm = N.dsp_masks(seg, masker)
        hf, pf = float(hm.mean()), float(pm.mean())
        stats.append((pf / (hf + pf + 1e-8), hf / (hf + pf + 1e-8)))
        t = torch.from_numpy(seg).unsqueeze(0).to(N.DEVICE)
        with torch.no_grad():
            est = separator(t)
        wola_v.push(est[0, 0].mean(0).cpu().numpy())   # raw NN vocals
        wola_i.push(est[0, 1].mean(0).cpu().numpy())   # raw NN instrumentals
    vocals = wola_v.flush()[:n_sample]
    instrs = wola_i.flush()[:n_sample]
    nch = int(np.ceil(n_sample / CHUNK_LEN))
    perc = np.repeat([s[0] for s in stats], cph)[:nch]
    harm = np.repeat([s[1] for s in stats], cph)[:nch]
    return vocals, instrs, perc, harm


def main():
    mp3s = sorted(Path("input").glob("*.mp3"))
    print("=" * 78)
    print("  NOISE SENSITIVITY — output quality vs input noise, EEP vs UEP")
    print(f"  Noise levels: clean, {[f'{d}dB' for d in NOISE_LEVELS if d]}  "
          f"|  FEC: {LOSS_RATE*100:.0f}% {CHANNEL} {PLACEMENT}")
    print("=" * 78)

    print(f"  Loading Open-Unmix (umxhq) on {N.DEVICE}...")
    separator = openunmix.umxhq(targets=["vocals"], device=N.DEVICE, residual=True)
    separator = separator.to(N.DEVICE).eval()
    visqol = F.make_visqol_pool()
    cdpam  = F.make_cdpam()

    for mp3 in mp3s:
        audio = N.load_stereo(mp3)
        n = audio.shape[1]
        print(f"\n{'='*78}\n  {mp3.stem}\n{'='*78}")

        # Clean reference separation
        ref_v, ref_i, _, _ = separate_raw(audio, separator, n)
        ref_vw = F.score_window(ref_v)
        ref_iw = F.score_window(ref_i)
        ref_mw = F.score_window(ref_v + ref_i)
        cd_anchor = F.cdpam_distance(cdpam, ref_mw, np.zeros_like(ref_mw))

        print(f"\n  {'Noise':>7}  {'Mode':<5}  {'SDR-V':>6}  {'SDR-I':>6}  "
              f"{'CDPAM':>6}  {'ViSQOL':>7}")
        print(f"  {'-'*52}")

        for lvl in NOISE_LEVELS:
            noisy = audio if lvl is None else N.add_noise(audio, lvl)
            v, i, perc, harm = separate_raw(noisy, separator, n)
            vch, orig = F.chunk_audio(v, CHUNK_LEN)
            ich, _    = F.chunk_audio(i, CHUNK_LEN)
            nch = min(len(vch), len(perc), len(harm))
            vch, ich = vch[:nch], ich[:nch]
            perc, harm = perc[:nch], harm[:nch]
            conf = np.array([F.vocal_confidence_score(c) for c in vch])
            sil  = np.array([F.is_silent(vch[k]) & F.is_silent(ich[k]) for k in range(nch)])
            rng = np.random.default_rng(42)

            vq_jobs = []
            rowbuf = []
            for mode in ("eep", "uep"):
                vo, io, vr, ir = F.simulate_fec(
                    vch, ich, conf, sil, perc, harm, LOSS_RATE, rng, mode,
                    channel=CHANNEL, placement=PLACEMENT)
                vf = vo.flatten()[:orig]
                ff = io.flatten()[:orig]
                mix = vf + ff
                sdr_v = F.si_sdr_100(ref_vw, F.score_window(vf))
                sdr_i = F.si_sdr_100(ref_iw, F.score_window(ff))
                cd = F.cdpam_score(F.cdpam_distance(cdpam, ref_mw, F.score_window(mix)), cd_anchor)
                vq_jobs.append((ref_mw, F.score_window(mix)))
                rowbuf.append([mode, sdr_v, sdr_i, cd, None])
            vqs = visqol.map(F._visqol_measure, vq_jobs) if visqol else [None, None]
            lbl = "clean" if lvl is None else f"{lvl}dB"
            for r, vq in zip(rowbuf, vqs):
                r[4] = vq
                print(f"  {lbl:>7}  {r[0].upper():<5}  {F.fmt_score(r[1]):>6}  "
                      f"{F.fmt_score(r[2]):>6}  {F.fmt_score(r[3]):>6}  {F.fmt_score(r[4]):>7}")
            print()

    if visqol:
        visqol.close(); visqol.join()
    print("=" * 78)
    print("  Done. Reference = clean-input separation; scores capture noise + loss.")
    print("=" * 78)


if __name__ == "__main__":
    main()
