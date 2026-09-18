"""
Wall-clock cost of a single full-length FFT evaluation (all bins, the
fair baseline since harmonics are then free reads) against the coprime
estimator's combined location + profile-readout cost, across four
record lengths built from coprime pairs (2^k-1, 2^k).

Run from the same directory as twostage.py, coprime.py, and fold_core.py.
"""
import time
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, verify_candidate
from fold_core import zpad_fft_full

lags = [4, 8, 12, 16, 20, 24]
sps = 24
H = 5
Bprof = 4096


def profile_via_coprime(x, q0, N, B, H):
    return np.array([verify_candidate(x, lags, (m * q0) % N / N, 0, B)
                      for m in range(1, H + 1)])


DEFAULT_CONFIGS = [(65280, 255, 256), (261632, 511, 512),
                   (1047552, 1023, 1024), (4192256, 2047, 2048)]


def run_cost_sweep(configs=DEFAULT_CONFIGS, reps=7):
    """Runs the timing sweep and returns a list of dicts, one per config:
    {N, M1, M2, baseline_ms, location_ms, readout_ms, total_ms, speedup}.
    REPS>=10 is recommended on a noisy/shared machine; the default of 7
    is a compromise between wall-clock time and jitter for routine use.
    """
    results = []
    for N, M1, M2 in configs:
        rng = np.random.default_rng(1)
        x = make_signal_exact_N("QPSK", "rect", sps, N, 15.0, rng)

        # fair baseline: one full N-length FFT per lag, using every sample of
        # the window (S = {0,...,N-1}), not a decimated subset -- harmonics
        # are then free reads from the already-computed spectrum
        T_idx = np.arange(N)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for tau in lags:
                zpad_fft_full(x, tau, T_idx, N)
            times.append(time.perf_counter() - t0)
        t_base = np.median(times)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
            times.append(time.perf_counter() - t0)
        t_loc = np.median(times)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            profile_via_coprime(x, q0, N, Bprof, H)
            times.append(time.perf_counter() - t0)
        t_prof = np.median(times)

        t_total = t_loc + t_prof
        results.append(dict(N=N, M1=M1, M2=M2, baseline_ms=1000 * t_base,
                             location_ms=1000 * t_loc, readout_ms=1000 * t_prof,
                             total_ms=1000 * t_total, speedup=t_base / t_total))
    return results


if __name__ == "__main__":
    print(f"{'N':>10}{'(M1,M2)':>12}{'baseline(ms)':>14}{'location(ms)':>14}"
          f"{'readout(ms)':>13}{'total(ms)':>11}{'speedup':>9}")
    for r in run_cost_sweep():
        pair_str = f"({r['M1']},{r['M2']})"
        print(f"{r['N']:10d}{pair_str:>12}{r['baseline_ms']:14.2f}"
              f"{r['location_ms']:14.2f}{r['readout_ms']:13.2f}{r['total_ms']:11.2f}"
              f"{r['speedup']:9.1f}")
