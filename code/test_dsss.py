"""
DSSS-BPSK validation case: chip rate 0.25, spreading gain 31, matching the
S3CA validation configuration.

This is a genuine short-code DSSS signal, not the plain rectangular-chip
model used elsewhere in this codebase: each data bit is spread by a
fixed-length pseudorandom +-1 chip sequence (reused every data bit, i.e.
a "short code"), then chip-shaped with a rectangular pulse. Reusing the
same code every bit is what gives the signal a strong cyclic feature at
the DATA rate (chip_rate / spreading_gain) in addition to the chip-rate
feature any rectangular-chip signal has -- this is the classical
short-code DSSS cyclostationarity result.

NOTE: this script tests location and profile readout only. It does NOT
compare against dense/sparse SCD reconstruction or against the S3CA
algorithm directly -- neither a dense SCD reconstructor nor an S3CA
comparison harness for this specific signal exists in this script, and
building either is a separate undertaking from the coprime estimator
this script validates.
"""
import numpy as np
from coprime import coprime_alpha0, verify_candidate

lags = [4, 8, 12, 16, 20, 24]


def mseq_length31(seed_state=0b10101):
    """Maximal-length sequence from a 5-bit LFSR, period 2^5-1=31, taps at
    positions 5,3 (x^5+x^3+1, a standard primitive polynomial for degree 5).
    Returns +-1 chips."""
    state = seed_state
    bits = []
    for _ in range(31):
        bits.append(state & 1)
        fb = ((state >> 4) ^ (state >> 2)) & 1   # taps for x^5+x^3+1
        state = ((state << 1) | fb) & 0b11111
    return 1.0 - 2.0 * np.array(bits)             # 0/1 -> +1/-1


PN_CODE = mseq_length31()


def make_dsss_signal(chip_rate, spreading_gain, N, snr_db, rng, guard=64, code=None):
    """Short-code BPSK DSSS: data bits at chip_rate/spreading_gain, each
    spread by the SAME length-`spreading_gain` +-1 chip sequence, chip
    period 1/chip_rate samples, rectangular chip shaping."""
    sps_chip = round(1.0 / chip_rate)
    sym_period = sps_chip * spreading_gain
    nbits = (N + guard) // sym_period + 4
    if code is None:
        code = PN_CODE                                            # fixed m-sequence, not re-drawn per trial
    bits = rng.choice([-1.0, 1.0], size=nbits)
    chips = (bits[:, None] * code[None, :]).ravel()               # length nbits*spreading_gain
    x = np.repeat(chips, sps_chip).astype(complex)                # rectangular chip shaping
    x = x[: N + guard]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    p_n = 10 ** (-snr_db / 10)
    noise = np.sqrt(p_n / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise, sym_period, sps_chip


if __name__ == "__main__":
    # N must be a multiple of the true symbol period (chip_rate's reciprocal
    # times spreading_gain) for the fundamental to land exactly on a native
    # bin. With spreading_gain=31 (prime) and chip period 4 samples, the
    # symbol period is 124 -- NOT a divisor of 65280=256*255, which was the
    # actual cause of the earlier "wrong harmonic" results at this N: the
    # true frequency was never on the native grid to begin with. M1=248
    # (=8*31, carrying the spreading-gain factor) and M2=249 (=3*83) are
    # coprime and their product is a multiple of 124.
    N = 61752
    M1, M2 = 248, 249
    chip_rate = 0.25
    spreading_gain = 31
    SNR = 10.0

    rng0 = np.random.default_rng(0)
    _, sym_period, sps_chip = make_dsss_signal(chip_rate, spreading_gain, N, SNR, rng0)
    a0_data = 1.0 / sym_period       # the code-period (data) rate the harmonic-comb model targets
    a0_chip = chip_rate               # the chip rate itself, also cyclostationary
    print(f"chip period = {sps_chip} samples, symbol (code) period = {sym_period} samples")
    print(f"data rate alpha0 = {a0_data:.6f}, chip rate = {a0_chip:.3f}\n")

    ok_data = ok_chip = 0
    ntrials = 20
    for tr in range(ntrials):
        rng = np.random.default_rng(9000 + tr)
        x, _, _ = make_dsss_signal(chip_rate, spreading_gain, N, SNR, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
        if ah is not None:
            if abs(ah - a0_data) < 0.5 / N:
                ok_data += 1
            elif abs(ah - a0_chip) < 0.5 / N:
                ok_chip += 1
    print(f"locked onto data rate: {ok_data}/{ntrials}")
    print(f"locked onto chip rate: {ok_chip}/{ntrials}")
    print(f"(coprime_alpha0 now applies divisor-consistency resolution -- see "
          f"resolve_harmonic_order in coprime.py -- which prefers the smaller of "
          f"two genuine cyclic features related by an integer ratio; the chip "
          f"rate here is exactly {round(chip_rate/a0_data)}x the data rate, so it "
          f"is consistently resolved down to the data rate rather than reported "
          f"directly)")

    # profile at harmonics of whichever rate was found in a representative trial
    rng = np.random.default_rng(9000)
    x, _, _ = make_dsss_signal(chip_rate, spreading_gain, N, SNR, rng)
    ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
    print(f"\nrepresentative trial: locked onto alpha0={ah}")
    for m in range(1, 6):
        qm = (m * q0) % N
        Pfull = verify_candidate(x, lags, qm / N, 0, N)
        print(f"  m={m}  alpha={qm/N:.5f}  P_full={Pfull:.4f}")
