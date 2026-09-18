import numpy as np
from fold_core import fine_fft, coarse_fft
from signal_model import make_signal, local_floor  # signal model + noise-floor estimate


def make_signal_exact_N(mod, pulse, sps, N, snr_db, rng, fc=0.0, guard=64):
    """make_signal but truncated/padded so indices 0..N-1+guard are valid,
    letting N stay a clean power of two regardless of sps (needed for exact
    decimation) while lag offsets never overrun the array."""
    nsym = (N + guard) // sps + 8
    x = make_signal(mod, pulse, sps, nsym, snr_db, rng, fc=fc)
    return x[: N + guard]



def fine_statistic(x, lags, t0, D, M):
    acc = np.zeros(M)
    for tau in lags:
        Y = fine_fft(x, tau, t0, D, M)
        acc += np.abs(Y / M) ** 2
    return np.sqrt(acc)


def coarse_statistic(x, lags, start, B):
    acc = np.zeros(B)
    for tau in lags:
        Y = coarse_fft(x, tau, start, B)
        acc += np.abs(Y / B) ** 2
    return np.sqrt(acc)


def peak_candidates(Tfold, M, gate=3.0, n=4, half=8, guard=2, pmin=1):
    """Local maxima of Tfold above a median-floor gate, excluding p=0."""
    floors = np.array([local_floor(Tfold, M, p / M, half=half, guard=guard)
                        for p in range(M)])
    cands = []
    for p in range(pmin, M // 2 + 1):
        if Tfold[p] < gate * floors[p]:
            continue
        if Tfold[p] >= Tfold[p - 1] and Tfold[p] >= Tfold[(p + 1) % M]:
            cands.append(p)
    cands.sort(key=lambda p: -Tfold[p])
    return cands[:n]


def resolve_coset(p, N, M, D, Tcoarse, B):
    """Pick j in 0..D-1 maximizing Tcoarse at the bin nearest alpha_j=(p+jM)/N."""
    best_j, best_s = 0, -1.0
    for j in range(D):
        alpha_j = (p + j * M) / N
        b = int(round(alpha_j * B)) % B
        s = Tcoarse[b]
        if s > best_s:
            best_s, best_j = s, j
    return best_j, best_s


def two_stage_alpha0(x, lags, N, M, t0=0, coarse_start=0, B=None,
                      fine_gate=3.0, sym_period_floor=170.0, alpha_guess=None,
                      jitter=None, rng=None):
    """Returns (alpha_hat, p, j, diag) for the strongest fine candidate.

    B must satisfy two independent constraints: a resolution constraint
    B >= ~2D so adjacent cosets (spaced 1/D in alpha) land in different
    coarse bins, and a reliability constraint B*alpha0 >= sym_period_floor
    so the coarse (undecimated, single-shot) stage has enough observed
    symbol periods to detect the peak at all. The second is empirically
    calibrated (see calibrate_sym_period_floor) and is independent of M;
    for large M it is usually the binding constraint, not the resolution
    one -- pass alpha_guess (a rough expected rate) to size B correctly
    without needing the true alpha0.
    """
    D = N // M
    if B is None:
        res_floor = 8 * D
        rate_floor = sym_period_floor / alpha_guess if alpha_guess else res_floor
        B = int(max(res_floor, rate_floor))
        B = min(B, N)
    if jitter is not None:
        base = t0 + D * np.arange(M)
        idx = base + rng.integers(-jitter, jitter + 1, size=M)
        idx = np.clip(idx, 0, N - 1)
        acc = np.zeros(M)
        for tau in lags:
            z = x[idx + tau] * np.conj(x[idx])
            acc += np.abs(np.fft.fft(z) / M) ** 2
        Tfold = np.sqrt(acc)
    else:
        Tfold = fine_statistic(x, lags, t0, D, M)
    Tcoarse = coarse_statistic(x, lags, coarse_start, B)
    cands = peak_candidates(Tfold, M, gate=fine_gate)
    if not cands:
        return None, None, None, dict(Tfold=Tfold, Tcoarse=Tcoarse, B=B)
    p = cands[0]
    p_mirror = (M - p) % M
    j1, s1 = resolve_coset(p, N, M, D, Tcoarse, B)
    j2, s2 = resolve_coset(p_mirror, N, M, D, Tcoarse, B)
    if s1 >= s2:
        alpha_hat = (p + j1 * M) / N
    else:
        alpha_hat = (p_mirror + j2 * M) / N
    if alpha_hat > 0.5:
        alpha_hat = 1.0 - alpha_hat            # canonicalize +-alpha0 mirror
    return alpha_hat, p, (j1 if s1 >= s2 else j2), dict(
        Tfold=Tfold, Tcoarse=Tcoarse, cands=cands, B=B)
