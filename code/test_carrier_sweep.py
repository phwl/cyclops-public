"""
Carrier-frequency invariance test: profile error vs. carrier offset f_c,
checking whether recovery degrades near Nyquist.

The coprime estimator has no channelizer at any stage -- location uses
the non-conjugate cyclic autocorrelation directly, and the harmonic
readout does too -- so carrier phase should cancel out of the
non-conjugate product entirely and recovery/profile accuracy should be
INDEPENDENT of f_c, including near Nyquist. This script tests that
prediction rather than assuming it.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, verify_candidate

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]
sps = 20               # rate 0.05, a representative mid-range symbol rate
a0 = 1.0 / sps
H = 5
Bprof = 4096
SNR = 15.0
NTRIALS = 20


def off_comb_floor(x, q0, B, n_probes=6):
    probes = (np.arange(1, n_probes + 1) + 0.37) / (2 * n_probes)
    return np.median([verify_candidate(x, lags, p, 0, B) for p in probes])


def profile_via_coprime(x, q0, B, H):
    return np.array([verify_candidate(x, lags, (m * q0) % N / N, 0, B)
                      for m in range(1, H + 1)])


if __name__ == "__main__":
    print(f"{'fc':>7}{'recovery':>10}{'mean rel err':>13}{'profile err':>13}{'m used':>8}")
    for fc in [-0.49, -0.4, -0.25, 0.0, 0.25, 0.4, 0.49]:
        ok = 0
        rel_alpha = []
        prof_errs = []
        m_used = []
        for tr in range(NTRIALS):
            rng = np.random.default_rng(7000 + tr)
            x = make_signal_exact_N("QPSK", "rect", sps, N, SNR, rng, fc=fc)
            ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
            if ah is None:
                continue
            if abs(ah - a0) < 0.5 / N:
                ok += 1
            rel_alpha.append(abs(ah - a0) / a0)
            Pfull = profile_via_coprime(x, q0, N, H)
            Phat = profile_via_coprime(x, q0, Bprof, H)
            floor = off_comb_floor(x, q0, Bprof)
            populated = Pfull > 15 * floor
            if not populated.any():
                continue
            e = np.abs(Phat[populated] - Pfull[populated]) / Pfull[populated]
            prof_errs.append(np.mean(e))
            m_used.append(int(populated.sum()))
        print(f"{fc:7.2f}{ok/NTRIALS:10.2f}"
              f"{np.mean(rel_alpha) if rel_alpha else float('nan'):13.1e}"
              f"{np.mean(prof_errs) if prof_errs else float('nan'):13.3f}"
              f"{np.mean(m_used) if m_used else 0:8.1f}")
