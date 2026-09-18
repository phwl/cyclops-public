"""
Single-signal correctness sweep for the coprime estimator:
  - single-signal recovery rate across several symbol rates
  - degenerate-branch recovery (one of M1, M2 divides q_true exactly)
    compared against a single (non-coprime) decimation under the same
    degeneracy.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N, two_stage_alpha0
from coprime import coprime_alpha0

N = 65280           # = 255 * 256
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]


def recovery_rate(sps, ntrials=30, snr_db=15.0, seed0=300):
    a0 = 1.0 / sps
    ok = 0
    for tr in range(ntrials):
        rng = np.random.default_rng(seed0 + tr)
        x = make_signal_exact_N("QPSK", "rect", sps, N, snr_db, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
        ok += (ah is not None and abs(ah - a0) < 0.5 / N)
    return ok / ntrials


if __name__ == "__main__":
    print("=== single-signal correctness ===")
    for sps in [24, 30, 40, 50]:
        print(f"  sps={sps:3d}  alpha0={1/sps:.5f}  recovery={recovery_rate(sps):.2f}")

    print("\n=== degenerate-branch recovery ===")
    print("  q_true is divisible by exactly one of M1=255, M2=256 for each rate below")
    for sps in [15, 16, 32, 51]:
        a0 = 1.0 / sps
        q_true = round(a0 * N)
        deg = "M1" if q_true % M1 == 0 else ("M2" if q_true % M2 == 0 else "none")
        print(f"  sps={sps:3d}  degenerate branch={deg}  "
              f"recovery={recovery_rate(sps, seed0=400):.2f}")

    print("\n=== a single (non-coprime) decimation under the same degeneracy: fails outright ===")
    print("  uses two_stage_alpha0 (twostage.py), the single-decimation pipeline with")
    print("  its own coarse-block disambiguation -- not the coprime construction --")
    print("  so that the comparison is against a complete competing pipeline, not a")
    print("  single unverified gate check.")
    sps = 16   # alpha0*D is an integer for D=N/M with M any power of two here,
               # so the true fundamental aliases to residue 0 regardless of M chosen
    ok = 0
    ntrials = 20
    for tr in range(ntrials):
        rng = np.random.default_rng(900 + tr)
        x = make_signal_exact_N("QPSK", "rect", sps, N, 15.0, rng)
        ah, p, j, diag = two_stage_alpha0(x, lags, N, M=512, alpha_guess=1.0 / sps)
        ok += (ah is not None and abs(ah - 1.0 / sps) < 1.5 / N)
    print(f"  sps={sps}: single-decimation recovery = {ok/ntrials:.2f} "
          f"(expected to fail outright under this exact degeneracy)")
