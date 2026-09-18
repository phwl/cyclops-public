"""
Lag-set structural nulls: a lag tau contributes essentially zero cyclic
signal, regardless of SNR, whenever tau < gcd(D, sps), where D is a
branch's decimation stride and sps is the symbol period. This is a
different, exact mechanism from the "lag exceeds pulse width" dilution
characterized elsewhere (see test_rate_reliability.py): it is a hard,
structural null tied to the specific (stride, symbol period) pair, and
it can silence individual lags -- or, in the worst case, the entire
fixed lag set on one branch -- even for lags well inside the pulse
width.

Mechanism: a branch's decimated samples visit only sps/gcd(D,sps)
distinct phase positions within each symbol. If a lag tau is smaller
than gcd(D,sps), every visited phase keeps t and t+tau inside the same
symbol at every decimated sample, so the delay product never sees a
symbol boundary and carries no information about the symbol clock --
regardless of modulation, constant-modulus or not.

Because the two branches' strides D1=M2, D2=M1 are coprime, they have
different factor structure, so it is rare (though not impossible) for
the same sps to cripple both branches at once -- but as the sps=96
case below shows, "not completely dead" is not the same as "healthy":
the surviving branch can still be significantly weakened.

Run from the same directory as twostage.py, coprime.py, and fold_core.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from fold_core import fine_fft
from coprime import coprime_alpha0

N, M1, M2 = 65280, 255, 256
D1, D2 = N // M1, N // M2
LAGS = [4, 8, 12, 16, 20, 24]


def per_lag_magnitude(sps, mod="QPSK", snr_db=30.0, ntrials=15, branch=1, lags=LAGS, seed0=9700):
    """Mean |cyclic feature| at the true residue, per lag, for one branch.
    High SNR by default so the measurement reflects the true expected
    signal strength rather than noise. Returns a dict {tau: mean_magnitude}.
    """
    D = D1 if branch == 1 else D2
    M = M1 if branch == 1 else M2
    r_true = round(N / sps) % M
    out = {}
    for tau in lags:
        vals = []
        for tr in range(ntrials):
            rng = np.random.default_rng(seed0 + tr)
            x = make_signal_exact_N(mod, "rect", sps, N, snr_db, rng)
            Y = fine_fft(x, tau, 0, D, M)
            vals.append(np.abs(Y[r_true]) / M)
        out[tau] = float(np.mean(vals))
    return out


def gcd_table(sps_values, lags=LAGS):
    """Returns a list of dicts {sps, gcd_d1, gcd_d2, min_lag, at_risk_branch1,
    at_risk_branch2}, flagging sps values where gcd(D_i, sps) meets or
    exceeds every lag in the set (a fully-dead branch)."""
    rows = []
    for sps in sps_values:
        g1, g2 = np.gcd(D1, sps), np.gcd(D2, sps)
        rows.append(dict(sps=sps, gcd_d1=int(g1), gcd_d2=int(g2),
                          at_risk_branch1=bool(g1 > min(lags)),
                          at_risk_branch2=bool(g2 > min(lags)),
                          fully_dead_branch1=bool(g1 > max(lags)),
                          fully_dead_branch2=bool(g2 > max(lags))))
    return rows


def recovery_at_sps(sps, snr_db=15.0, ntrials=20, lags=LAGS):
    """Full-pipeline recovery rate at a given sps, for comparison against
    the per-lag/gcd diagnostics above."""
    ok = 0
    a0 = 1.0 / sps
    for tr in range(ntrials):
        rng = np.random.default_rng(9960 + tr)
        x = make_signal_exact_N("QPSK", "rect", sps, N, snr_db, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
        ok += (ah is not None and abs(ah - a0) < 0.5 / N)
    return ok / ntrials


if __name__ == "__main__":
    print("=== gcd(D_i, sps) risk table ===")
    for r in gcd_table([7, 8, 9, 11, 12, 23, 24, 48, 96]):
        flag = ""
        if r["fully_dead_branch1"] and r["fully_dead_branch2"]:
            flag = "  <-- BOTH branches fully dead across the whole lag set"
        elif r["fully_dead_branch1"] or r["fully_dead_branch2"]:
            flag = "  <-- one branch fully dead across the whole lag set"
        print(f"  sps={r['sps']:3d}  gcd(D1,sps)={r['gcd_d1']:3d}  "
              f"gcd(D2,sps)={r['gcd_d2']:3d}{flag}")

    print("\n=== per-lag magnitude, branch 1, sps=24 (gcd=8) ===")
    for tau, m in per_lag_magnitude(24, branch=1).items():
        print(f"  tau={tau:2d}: {m:.3f}" + ("  (dead: tau < gcd)" if tau < 8 else ""))

    print("\n=== per-lag magnitude, branch 1, sps=96 (gcd=32, whole set dead) ===")
    for tau, m in per_lag_magnitude(96, branch=1).items():
        print(f"  tau={tau:2d}: {m:.3f}")

    print("\n=== per-lag magnitude, branch 2, sps=96 (gcd=3, weakened not dead) ===")
    for tau, m in per_lag_magnitude(96, branch=2).items():
        print(f"  tau={tau:2d}: {m:.3f}")

    print("\n=== full-pipeline recovery, sps=96 vs. a 'friendly' sps=24 ===")
    print(f"  sps=24: {recovery_at_sps(24):.2f}")
    print(f"  sps=96: {recovery_at_sps(96):.2f}")
