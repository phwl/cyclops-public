"""
Recovery outcome vs. sub-bin offset from the nearest native bin: sweeps
a symbol rate that does not land exactly on alpha=q/N (a symbol clock
unrelated to the record length will not, in general), and classifies
each trial as exact (locates the nearest native bin), wild (confidently
reports a different bin), or no-detection.

Run from the same directory as offgrid_gen.py and coprime.py.
"""
import numpy as np
from offgrid_gen import make_offgrid_signal_exact_N
from coprime import coprime_alpha0

N, M1, M2 = 65280, 255, 256
lags = [4, 8, 12, 16, 20, 24]
Q0_BASE = 2720
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
NTRIALS = 60
SNR = 15.0


def run_offgrid_sweep(deltas=DELTAS, ntrials=NTRIALS, snr_db=SNR, seed0=80000):
    """Returns a list of dicts, one per delta:
    {delta, exact, wild, no_detect} (fractions over ntrials trials)."""
    results = []
    for delta in deltas:
        q0_true = Q0_BASE + delta
        L = N / q0_true
        exact = wild = none = 0
        for tr in range(ntrials):
            rng = np.random.default_rng(seed0 + tr)
            x = make_offgrid_signal_exact_N("QPSK", L, N, snr_db, rng)
            ah, q_hat, diag = coprime_alpha0(x, lags, N, M1, M2)
            if ah is None:
                none += 1
            elif abs(q_hat - q0_true) < 0.5001:
                exact += 1
            else:
                wild += 1
        results.append(dict(delta=delta, exact=exact / ntrials,
                             wild=wild / ntrials, no_detect=none / ntrials))
    return results


if __name__ == "__main__":
    print(f"{'delta':>7}{'exact':>8}{'wild':>8}{'no-detect':>11}")
    for r in run_offgrid_sweep():
        print(f"{r['delta']:7.2f}{r['exact']:8.2f}{r['wild']:8.2f}{r['no_detect']:11.2f}")
