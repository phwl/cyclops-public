"""
Recovery rate vs. symbol rate, rectangular chip shaping, across four
modulations.

N is fixed at 65280 (M1=255, M2=256) throughout, so the "aperture"
available here is set by the coprime pair rather than by a separately
tunable subsample count. Rates are swept via sps (samples/symbol); sps
values landing on a degenerate residue for either M1 or M2 are skipped,
since that failure mode is already characterized separately (see
test_coprime_correctness.py) and would otherwise contaminate a
rate-reliability curve with a second, unrelated effect.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]
SNR = 15.0
NTRIALS = 20
MODS = ["BPSK", "QPSK", "8PSK", "16QAM"]

# sps values spanning a wide rate range (0.005 to 0.12 cycles/sample),
# skipping any that degenerate either branch
SPS_VALUES = [8, 10, 12, 16, 20, 24, 30, 40, 50, 64, 80, 100, 130, 160, 200]


def is_degenerate(sps):
    q_true = round((1.0 / sps) * N)
    return q_true % M1 == 0 or q_true % M2 == 0


def run_rate_sweep(sps_values=SPS_VALUES, mods=MODS, ntrials=NTRIALS):
    """Returns a list of dicts, one per non-degenerate sps value:
    {sps, rate, recovery: {mod_name: rate, ...}}. Degenerate sps values
    (see module docstring) are skipped entirely, matching the printed
    sweep below.
    """
    results = []
    for sps in sps_values:
        if is_degenerate(sps):
            continue
        a0 = 1.0 / sps
        recovery = {}
        for mod in mods:
            ok = 0
            for tr in range(ntrials):
                rng = np.random.default_rng(8000 + tr)
                x = make_signal_exact_N(mod, "rect", sps, N, SNR, rng)
                ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
                ok += (ah is not None and abs(ah - a0) < 0.5 / N)
            recovery[mod] = ok / ntrials
        results.append(dict(sps=sps, rate=a0, recovery=recovery))
    return results


if __name__ == "__main__":
    print(f"{'sps':>5}{'rate':>8}" + "".join(f"{m:>9}" for m in MODS))
    results_by_sps = {r["sps"]: r for r in run_rate_sweep()}
    for sps in SPS_VALUES:
        if sps not in results_by_sps:
            print(f"{sps:5d}{1/sps:8.4f}  -- skipped (degenerate residue for this pair) --")
            continue
        r = results_by_sps[sps]
        row = [r["recovery"][m] for m in MODS]
        print(f"{r['sps']:5d}{r['rate']:8.4f}" + "".join(f"{v:9.2f}" for v in row))
