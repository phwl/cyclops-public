"""
Off-grid signal generator: real-valued symbol rate (not necessarily N/integer),
via rounded symbol-boundary quantization. Symbol k occupies samples
[round(k*L), round((k+1)*L)) for real symbol length L = 1/alpha0, so the
signal has a well-defined cyclic feature at alpha0 even when alpha0*N is
not an integer -- boundary rounding introduces small clock jitter (chip
lengths of floor(L) or ceil(L) samples) rather than true fractional-delay
interpolation, but this is adequate to test whether spectral leakage from
an off-grid alpha0 confuses the coprime location step.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from signal_model import symbols

def make_offgrid_signal(mod, L, nsym, snr_db, rng, fc=0.0):
    a = symbols(mod, nsym + 64, rng)
    boundaries = np.round(np.arange(nsym + 65) * L).astype(int)
    total_len = boundaries[-1]
    x_up = np.zeros(total_len, dtype=complex)
    for k in range(nsym + 64):
        x_up[boundaries[k]:boundaries[k+1]] = a[k]
    skip = boundaries[32]
    end = boundaries[32 + nsym]
    x = x_up[skip:end]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    if fc:
        x = x * np.exp(2j * np.pi * fc * np.arange(len(x)))
    p_n = 10 ** (-snr_db / 10)
    x = x + np.sqrt(p_n / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x

def make_offgrid_signal_exact_N(mod, L, N, snr_db, rng, fc=0.0, guard=64):
    nsym = int((N + guard) / L) + 8
    x = make_offgrid_signal(mod, L, nsym, snr_db, rng, fc=fc)
    return x[: N + guard]
