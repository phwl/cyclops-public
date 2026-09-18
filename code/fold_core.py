"""
O(L M log M) alpha-profile estimator.

Stage 1 (fine): uniform-stride subsample, one length-M FFT per lag.
    Folds the cycle-frequency axis exactly: |R_hat^{q/N}(tau)| depends only
    on q mod M.
Stage 2 (coarse): short contiguous (undecimated) block, one length-B FFT
    per lag, B ~ few * D where D = N/M. Resolves which of the D aliased
    cosets is correct.

Part 1 of this file verifies the folding identity numerically against the
brute-force zero-padded-FFT evaluation used in the current O(LN log N) draft.
"""
import numpy as np

RNG = np.random.default_rng(20260909)


def zpad_fft_full(x, tau, T_idx, N):
    """Current method: zero-fill z_tau at T_idx, one length-N FFT."""
    z = np.zeros(N, dtype=complex)
    z[T_idx] = x[T_idx + tau] * np.conj(x[T_idx])
    return np.fft.fft(z)


def fine_fft(x, tau, t0, D, M):
    """Two-stage fine step: uniform stride D, length-M FFT, no zero-fill."""
    idx = t0 + D * np.arange(M)
    z = x[idx + tau] * np.conj(x[idx])
    return np.fft.fft(z)  # Y_tau(p), p = 0..M-1


def coarse_fft(x, tau, start, B):
    """Two-stage coarse step: B contiguous samples, length-B FFT."""
    idx = start + np.arange(B)
    z = x[idx + tau] * np.conj(x[idx])
    return np.fft.fft(z)


# ------------------------------------------------------------ 1. verify identity
if __name__ == "__main__":
    N, M = 4096, 64
    D = N // M
    t0, tau = 3, 7
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(N + 32) + 1j * rng.standard_normal(N + 32))

    T_idx = t0 + D * np.arange(M)
    full = zpad_fft_full(x, tau, T_idx, N)          # length N
    Y = fine_fft(x, tau, t0, D, M)                  # length M

    q = np.arange(N)
    predicted = np.exp(-2j * np.pi * q * t0 / N) * Y[q % M]
    err = np.max(np.abs(full - predicted)) / np.max(np.abs(full))
    print(f"folding identity: max relative error over all N={N} bins = {err:.3e}")
    assert err < 1e-10, "folding identity failed"
    print("folding identity verified to float precision.\n")
