import time
import numpy as np
from s3ca import dense_ssca, s3ca as s3ca_fn, _centered
from twostage import make_signal_exact_N
from coprime import coprime_alpha0, verify_candidate

Np = 32
kappa = 8
sps = 24
a0 = 1.0 / sps
lags = [4, 8, 12, 16, 20, 24]
H = 5


def profile_at_alpha_dense(S, k_idx, N, Np, alpha_target):
    best = 0.0
    for ki, k in enumerate(k_idx):
        fk = k / Np
        qc = round((alpha_target - fk) * N)
        if qc < -(N // 2) or qc >= N - N // 2:
            continue
        row = qc + N // 2
        val = abs(S[row, ki])
        if val > best:
            best = val
    return best


def profile_at_alpha_sparse(result, N, alpha_target, tol_bins=1.0):
    tol = tol_bins / N
    mask = np.abs(result.alpha - alpha_target) <= tol
    if not mask.any():
        return 0.0
    return np.max(np.abs(result.value[mask]))


DEFAULT_CONFIGS = [(65280, 255, 256), (261632, 511, 512), (1047552, 1023, 1024)]


def run_s3ca_comparison(configs=DEFAULT_CONFIGS):
    """Runs the S3CA vs. coprime timing/accuracy comparison and returns a
    list of dicts, one per config:
    {N, M1, M2, s3ca_dense_ms, s3ca_sfft1_ms, coprime_ms, speedup,
     harmonics: [{m, dense, sfft1, rel_err}, ...]}
    """
    results = []
    for N, M1, M2 in configs:
        rng = np.random.default_rng(1)
        x_guard = make_signal_exact_N("QPSK", "rect", sps, N, 15.0, rng)  # N+64 samples
        x = x_guard[:N]

        t0 = time.perf_counter()
        S, f_grid, alpha_grid = dense_ssca(x, Np)
        t_dense = time.perf_counter() - t0
        k_idx = _centered(Np)

        t0 = time.perf_counter()
        res = s3ca_fn(x, Np, kappa, seed=0)
        t_s3ca = time.perf_counter() - t0

        t0 = time.perf_counter()
        ah, q0, diag = coprime_alpha0(x_guard, lags, N, M1, M2)
        for m in range(1, H + 1):
            verify_candidate(x_guard, lags, (m * q0) % N / N, 0, 4096)
        t_ours = time.perf_counter() - t0

        harmonics = []
        for m in range(1, H + 1):
            vd = profile_at_alpha_dense(S, k_idx, N, Np, m * a0)
            vs = profile_at_alpha_sparse(res, N, m * a0)
            err = abs(vs - vd) / vd if vd else float("nan")
            harmonics.append(dict(m=m, dense=vd, sfft1=vs, rel_err=err))

        results.append(dict(
            N=N, M1=M1, M2=M2,
            s3ca_dense_ms=1000 * t_dense, s3ca_sfft1_ms=1000 * t_s3ca,
            coprime_ms=1000 * t_ours, speedup=t_s3ca / t_ours,
            alpha_correct=(ah is not None and abs(ah - a0) < 0.5 / N),
            harmonics=harmonics))
    return results


if __name__ == "__main__":
    for r in run_s3ca_comparison():
        N = r["N"]
        print(f"\n=== N={N} ({N/1e6:.2f}M samples), Np={Np}, kappa={kappa} ===")
        print(f"  timing (ms): S3CA dense={r['s3ca_dense_ms']:.1f}  "
              f"S3CA sfft1={r['s3ca_sfft1_ms']:.1f}  our coprime={r['coprime_ms']:.1f}")
        print(f"  our coprime vs S3CA sfft1 (same task, alpha profile only): "
              f"{r['speedup']:.1f}x faster")
        print(f"  alpha_hat correct: {r['alpha_correct']}")
        print(f"  {'m':>3}{'dense_ssca':>14}{'s3ca(sfft1)':>14}{'rel.err':>10}")
        for h in r["harmonics"]:
            print(f"  {h['m']:3d}{h['dense']:14.1f}{h['sfft1']:14.1f}{h['rel_err']:10.3f}")
