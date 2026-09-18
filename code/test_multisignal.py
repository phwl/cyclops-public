"""
Two-signal interference test: with two independent rectangular-chip
signals present at unrelated symbol rates, measures how often the
coprime estimator locks onto a true rate, a spurious (non-existent)
frequency, or declines to answer.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]

sps_a, sps_b = 24, 40
a0_a, a0_b = 1.0 / sps_a, 1.0 / sps_b


def two_signal(sps_a, sps_b, N, snr_db, rng, power_ratio=1.0):
    xa = make_signal_exact_N("QPSK", "rect", sps_a, N, 60.0, rng)  # ~noiseless
    xb = make_signal_exact_N("QPSK", "rect", sps_b, N, 60.0, rng)
    xa = xa / np.sqrt(np.mean(np.abs(xa) ** 2))
    xb = xb / np.sqrt(np.mean(np.abs(xb) ** 2)) * np.sqrt(power_ratio)
    mix = xa[:N] + xb[:N]
    mix = mix / np.sqrt(np.mean(np.abs(mix) ** 2))
    p_n = 10 ** (-snr_db / 10)
    noise = np.sqrt(p_n / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    out = mix + noise
    return np.r_[out, out[:64]]   # guard samples for lag indexing


if __name__ == "__main__":
    print(f"true rates: a0_a={a0_a:.5f} (sps={sps_a}), a0_b={a0_b:.5f} (sps={sps_b})\n")
    for pr in [1.0, 0.3, 0.1]:
        hit = spur = none = 0
        ntr = 60
        for tr in range(ntr):
            rng = np.random.default_rng(3000 + tr)
            x = two_signal(sps_a, sps_b, N, 15.0, rng, power_ratio=pr)
            ah, q, diag = coprime_alpha0(x, lags, N, M1, M2)
            if ah is None:
                none += 1
            elif abs(ah - a0_a) < 0.5 / N or abs(ah - a0_b) < 0.5 / N:
                hit += 1
            else:
                spur += 1
        print(f"power ratio b/a={pr:.1f}:  correct={hit}/{ntr}  "
              f"spurious={spur}/{ntr}  no-detection={none}/{ntr}")
