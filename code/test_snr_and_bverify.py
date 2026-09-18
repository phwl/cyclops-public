"""
Recovery rate vs. SNR and vs. verification block length:
  - recovery rate vs. SNR at fixed verification block length B
  - recovery rate vs. B at several SNRs (shows the tie-break is
    essentially free -- flat across B from 16 to 4096 samples)

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]
sps = 24
a0 = 1.0 / sps


def recovery_rate(snr_db, B_verify, ntrials=60, seed0=1000):
    ok = 0
    for tr in range(ntrials):
        rng = np.random.default_rng(seed0 + tr)
        x = make_signal_exact_N("QPSK", "rect", sps, N, snr_db, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2, B_verify=B_verify)
        ok += (ah is not None and abs(ah - a0) < 0.5 / N)
    return ok / ntrials


if __name__ == "__main__":
    print("=== recovery vs. SNR, B_verify=1024 fixed ===")
    for snr in [10, 6, 4, 2, 0, -2]:
        print(f"  SNR={snr:3d} dB  recovery={recovery_rate(snr, 1024, seed0=2000):.2f}")

    print("\n=== recovery vs. B_verify, insensitivity check ===")
    for Bv in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        row = [recovery_rate(snr, Bv, ntrials=60) for snr in [15.0, 10.0, 8.0]]
        print(f"  B_verify={Bv:5d}  15dB={row[0]:.2f}  10dB={row[1]:.2f}  8dB={row[2]:.2f}")
