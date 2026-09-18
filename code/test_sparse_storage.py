"""
Storage requirement: the pipeline reads only O(sqrt(N) + B) of the N
sample positions, not all N.

Every position the estimator touches is fixed in advance by (M1, M2,
lags, B) -- it does not depend on anything discovered at run time:

  - branch i reads x(s*D_i) and x(s*D_i + tau) for each lag tau,
    giving (L+1)*M_i positions per branch;
  - verification, harmonic-order resolution, and profile readout share
    one contiguous block of B samples. The frequencies scored on that
    block are chosen at run time, but its *location* is not, so it can
    be retained from the start.

A receiver that selects samples as it acquires therefore never needs to
hold the whole record. This is checked two ways below: by counting the
distinct positions actually required, and -- more stringently -- by
overwriting every non-retained sample with NaN and confirming the
unmodified estimator still recovers the correct answer. NaN is used
deliberately rather than zero: any access to a discarded position
propagates and poisons the result, so a passing run is evidence that
nothing outside the retained set was touched.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0

LAGS = [4, 8, 12, 16, 20, 24]
B = 4096
CONFIGS = [(65280, 255, 256), (261632, 511, 512),
           (1047552, 1023, 1024), (4192256, 2047, 2048)]


def retained_positions(N, M1, M2, lags=LAGS, B=B):
    """The exact set of sample indices the pipeline can read, fixed in
    advance from (M1, M2, lags, B)."""
    D1, D2 = N // M1, N // M2
    pos = set()
    for M, D in [(M1, D1), (M2, D2)]:
        base = D * np.arange(M)
        pos.update(base.tolist())
        for tau in lags:
            pos.update((base + tau).tolist())
    pos.update(range(B))
    return pos


def storage_table(configs=CONFIGS, lags=LAGS, B=B):
    """Returns a list of dicts {N, M1, M2, retained, fraction, bound},
    where `bound` is the closed form (L+1)(M1+M2)+B."""
    rows = []
    for N, M1, M2 in configs:
        pos = retained_positions(N, M1, M2, lags, B)
        rows.append(dict(N=N, M1=M1, M2=M2, retained=len(pos),
                         fraction=len(pos) / N,
                         bound=(len(lags) + 1) * (M1 + M2) + B))
    return rows


def verify_sparse_recovery(N=65280, M1=255, M2=256, sps=24, snr_db=15.0,
                           lags=LAGS, B=B, ntrials=10):
    """Runs the unmodified estimator on a record whose non-retained
    samples have been overwritten with NaN. Returns (n_correct, ntrials).
    """
    a0 = 1.0 / sps
    keep = retained_positions(N, M1, M2, lags, B)
    ok = 0
    for tr in range(ntrials):
        rng = np.random.default_rng(4242 + tr)
        x_full = make_signal_exact_N("QPSK", "rect", sps, N, snr_db, rng)
        idx = np.fromiter((p for p in keep if p < len(x_full)), dtype=int)
        x_sparse = np.full_like(x_full, np.nan + 0j)
        x_sparse[idx] = x_full[idx]
        ah, q0, diag = coprime_alpha0(x_sparse, lags, N, M1, M2, B_verify=B)
        ok += (ah is not None and abs(ah - a0) < 0.5 / N)
    return ok, ntrials


if __name__ == "__main__":
    print("=== distinct sample positions required ===")
    print(f"{'N':>10}{'retained':>10}{'% of N':>9}{'closed form':>13}")
    for r in storage_table():
        print(f"{r['N']:10d}{r['retained']:10d}{100*r['fraction']:8.2f}%"
              f"{r['bound']:13d}")
    print("\n  (each 4x in N multiplies storage by ~2x, not 4x: O(sqrt(N)+B))")

    print("\n=== recovery with all non-retained samples set to NaN ===")
    ok, n = verify_sparse_recovery()
    print(f"  N=65280, QPSK sps=24, 15 dB: {ok}/{n} correct")
    print("  (any read outside the retained set would propagate NaN and fail)")
