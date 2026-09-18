"""
Signal model used throughout: linearly modulated complex baseband
signals (BPSK/QPSK/8PSK/16QAM) with rectangular or root-raised-cosine
chip shaping, additive complex Gaussian noise, and optional carrier
offset. Also provides local_floor, the local noise-floor estimate used
by the fine-stage peak search in twostage.py.
"""
import numpy as np


def rrc_taps(beta, sps, span=16):
    """Root-raised-cosine impulse response, unit energy."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    if beta == 0.0:
        h = np.sinc(t)
    else:
        h = np.empty_like(t)
        for i, ti in enumerate(t):
            if abs(ti) < 1e-12:
                h[i] = 1 - beta + 4 * beta / np.pi
            elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
                h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    - (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            else:
                h[i] = (
                    np.sin(np.pi * ti * (1 - beta))
                    + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def symbols(mod, n, rng):
    if mod == "BPSK":
        c = np.array([1, -1], dtype=complex)
    elif mod == "QPSK":
        c = np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))
    elif mod == "8PSK":
        c = np.exp(1j * np.pi / 4 * np.arange(8))
    elif mod == "16QAM":
        g = np.array([-3, -1, 1, 3])
        c = (g[:, None] + 1j * g[None, :]).ravel()
    else:
        raise ValueError(f"unknown modulation {mod!r}")
    c = c / np.sqrt(np.mean(np.abs(c) ** 2))
    return c[rng.integers(0, len(c), n)]


def make_signal(mod, pulse, sps, nsym, snr_db, rng, fc=0.0):
    """Linearly modulated complex baseband signal of length nsym*sps.
    pulse="rect" for rectangular chip shaping, or a float excess
    bandwidth beta in (0,1] for root-raised-cosine shaping."""
    a = symbols(mod, nsym + 64, rng)
    up = np.zeros((nsym + 64) * sps, dtype=complex)
    up[::sps] = a
    if pulse == "rect":
        h = np.ones(sps) / np.sqrt(sps)
    else:
        h = rrc_taps(pulse, sps)
    x = np.convolve(up, h)
    skip = 32 * sps
    x = x[skip: skip + nsym * sps]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    if fc:
        x = x * np.exp(2j * np.pi * fc * np.arange(len(x)))
    p_n = 10 ** (-snr_db / 10)
    x = x + np.sqrt(p_n / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x


def local_floor(T, N, alpha, half=400, guard=20):
    """Median of T in a window around alpha, excluding a guard band --
    a noise-floor estimate local to that frequency rather than global."""
    q = int(round(alpha * N))
    lo, hi = max(1, q - half), min(N // 2, q + half)
    band = np.r_[T[lo: max(lo, q - guard)], T[min(hi, q + guard): hi]]
    return np.median(band) if len(band) else np.nan
