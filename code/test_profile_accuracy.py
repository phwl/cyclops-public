"""
Profile-magnitude accuracy vs. readout block length: once the
fundamental is located, harmonic bins are known exactly, and profile
magnitude at each is read out with V(alpha) over a block of B
samples. This sweeps B and compares against the same statistic
evaluated over the entire record (B=N), which stands in for the value
the estimator would give if cost were no constraint.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, verify_candidate

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]
sps = 24
H = 5    # populated harmonics for this pulse shape/rate; the 6th sits at the
          # off-comb noise floor and is excluded by that same criterion


def profile_via_coprime(x, q0, B, H):
    return np.array([verify_candidate(x, lags, (m * q0) % N / N, 0, B)
                      for m in range(1, H + 1)])


if __name__ == "__main__":
    for Bprof in [256, 512, 1024, 2048, 4096]:
        rel_errs = []
        for tr in range(40):
            rng = np.random.default_rng(5000 + tr)
            x = make_signal_exact_N("QPSK", "rect", sps, N, 15.0, rng)
            ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
            if q0 is None:
                continue
            Phat = profile_via_coprime(x, q0, Bprof, H)
            Pfull = profile_via_coprime(x, q0, N, H)
            rel_errs.append(np.abs(Phat - Pfull) / Pfull)
        rel_errs = np.array(rel_errs)
        print(f"B={Bprof:5d}  mean relative profile error={np.mean(rel_errs):.3f}  "
              f"(n={len(rel_errs)} trials)")
