"""
Cross-modulation test: BPSK / QPSK / 8-PSK / 16-QAM at three symbol
rates, rectangular shaping, 15 dB SNR, reporting both fundamental-frequency
recovery and profile accuracy at the populated harmonics.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, verify_candidate

N = 65280
M1, M2 = 255, 256
lags = [4, 8, 12, 16, 20, 24]
H = 5
Bprof = 4096
SNR = 15.0
NTRIALS = 20

# three representative rates (0.02, 0.05, 0.10), expressed
# as the nearest integer sps this signal model requires
RATES = {0.10: 10, 0.05: 20, 0.02: 50}
MODS = ["BPSK", "QPSK", "8PSK", "16QAM"]


def profile_via_coprime(x, q0, B, H):
    return np.array([verify_candidate(x, lags, (m * q0) % N / N, 0, B)
                      for m in range(1, H + 1)])


def off_comb_floor(x, q0, B, n_probes=6):
    """Median V(alpha) at frequencies deliberately off any harmonic of q0,
    evaluated at the SAME block length B used for readout. The noise floor
    of V scales with B (shorter blocks average less noise down), so a
    harmonic must be judged 'populated' against the floor at the block
    length actually used to read it out, not against a longer reference
    block's floor -- a harmonic can sit comfortably above the full-record
    floor while remaining marginal at a much shorter B."""
    probes = (np.arange(1, n_probes + 1) + 0.37) / (2 * n_probes)
    return np.median([verify_candidate(x, lags, p, 0, B) for p in probes])


if __name__ == "__main__":
    print(f"{'Mod':<8}{'Rate':>6}{'recovery':>10}{'mean rel err':>10}{'profile err':>13}{'m used':>8}")
    for rate, sps in RATES.items():
        a0 = 1.0 / sps
        for mod in MODS:
            ok = 0
            rel_alpha = []
            prof_errs = []
            m_used_counts = []
            for tr in range(NTRIALS):
                rng = np.random.default_rng(6000 + tr)
                x = make_signal_exact_N(mod, "rect", sps, N, SNR, rng)
                ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
                if ah is None:
                    continue
                if abs(ah - a0) < 0.5 / N:
                    ok += 1
                rel_alpha.append(abs(ah - a0) / a0)
                Pfull = profile_via_coprime(x, q0, N, H)
                Phat = profile_via_coprime(x, q0, Bprof, H)
                floor_at_B = off_comb_floor(x, q0, Bprof)
                populated = Pfull > 15 * floor_at_B
                if not populated.any():
                    continue
                e = np.abs(Phat[populated] - Pfull[populated]) / Pfull[populated]
                prof_errs.append(np.mean(e))
                m_used_counts.append(int(populated.sum()))
            print(f"{mod:<8}{rate:>6.2f}{ok/NTRIALS:>10.2f}"
                  f"{np.mean(rel_alpha) if rel_alpha else float('nan'):>10.1e}"
                  f"{np.mean(prof_errs) if prof_errs else float('nan'):>13.3f}"
                  f"{np.mean(m_used_counts) if m_used_counts else 0:>8.1f}")
