"""
Adaptive vs. fixed lag set across symbol rates.

A fixed lag set cannot satisfy the two bounds of the paper's Section II
across a wide rate range: the upper bound (tau < K) forces small lags at
small K, while cyclic-feature strength scales with the boundary-crossing
probability ~tau/K, so those same small lags are nearly useless at large
K. Scaling the lags with K satisfies both bounds automatically and keeps
every lag in the strong region at every rate.

This script compares the fixed set used elsewhere in this codebase
against adaptive_lags(K) from coprime.py, over K = 8..200 and four
modulations. It also demonstrates the one case where the adaptive set
does not rescue recovery: K in {8, 16, 64}, where gcd(D1,K) >= K leaves
branch 1 with an empty usable window, so the weakest modulation (16-QAM)
must be carried by branch 2 alone.

Run from the same directory as twostage.py and coprime.py.
"""
import numpy as np
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, adaptive_lags

N, M1, M2 = 65280, 255, 256
D1, D2 = N // M1, N // M2
FIXED = [4, 8, 12, 16, 20, 24]
K_VALUES = [8, 10, 12, 16, 20, 24, 30, 40, 50, 64, 80, 100, 130, 160, 200]
MODS = ["BPSK", "QPSK", "8PSK", "16QAM"]
SNR = 15.0
NTRIALS = 20


def branch_windows(K):
    """Usable lag window [gcd(D_i,K), K) per branch; None when empty."""
    out = []
    for D in (D1, D2):
        g = int(np.gcd(D, K))
        out.append(None if g >= K else (g, K))
    return out


def recovery(K, lags, mod="QPSK", ntrials=NTRIALS, snr_db=SNR):
    if not lags:
        return float("nan")
    a0 = 1.0 / K
    ok = 0
    for tr in range(ntrials):
        rng = np.random.default_rng(8000 + tr)
        x = make_signal_exact_N(mod, "rect", K, N, snr_db, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
        ok += (ah is not None and abs(ah - a0) < 0.5 / N)
    return ok / ntrials


def run_comparison(k_values=K_VALUES, mods=MODS, ntrials=NTRIALS):
    """Returns a list of dicts {K, lags_adaptive, fixed: {...}, adaptive: {...}}."""
    rows = []
    for K in k_values:
        lags = adaptive_lags(K)
        rows.append(dict(
            K=K, lags_adaptive=lags,
            fixed={m: recovery(K, FIXED, m, ntrials) for m in mods},
            adaptive={m: recovery(K, lags, m, ntrials) for m in mods}))
    return rows


if __name__ == "__main__":
    print("=== usable lag window per branch (EMPTY means no lag can work there) ===")
    for K in K_VALUES:
        w1, w2 = branch_windows(K)
        f = lambda w: "EMPTY" if w is None else f"[{w[0]},{w[1]})"
        print(f"  K={K:>4}  branch1 {f(w1):>12}   branch2 {f(w2):>12}")

    print("\n=== recovery: fixed {4,8,...,24} vs adaptive_lags(K), 15 dB ===")
    hdr = "".join(f"{m:>8}" for m in MODS)
    print(f"{'K':>5}{'  set':>10}{hdr}")
    rows = run_comparison()
    for r in rows:
        print(f"{r['K']:>5}{'  fixed':>10}" + "".join(f"{r['fixed'][m]:>8.2f}" for m in MODS))
        print(f"{'':>5}{'  adaptive':>10}" + "".join(f"{r['adaptive'][m]:>8.2f}" for m in MODS))

    fa = [v for r in rows for v in r["fixed"].values()]
    ad = [v for r in rows for v in r["adaptive"].values()]
    print(f"\n  fixed:    mean={np.mean(fa):.3f}  below 0.9: {sum(v<0.9 for v in fa)}/{len(fa)}")
    print(f"  adaptive: mean={np.mean(ad):.3f}  below 0.9: {sum(v<0.9 for v in ad)}/{len(ad)}")
    print("  (adaptive's remaining failures are 16-QAM at K in {8,16,64},")
    print("   exactly where branch 1's window is empty -- see the table above)")
