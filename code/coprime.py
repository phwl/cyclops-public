"""
Coprime dual-rate disambiguation.

Two independent decimated fine stages, M1 and M2 samples with strides
D1=N/M1, D2=N/M2, gcd(M1,M2)=1, M1*M2=N. Each gives a residue of the true
bin q modulo its own M_i (exact, by the folding identity). CRT recombines
the two residues into q mod (M1*M2) = q mod N -- i.e. full native-grid
resolution, with no coarse contiguous block, no B, and no separate
aperture floor beyond what each fine stage already needs on its own.

Structural guarantee: simultaneous degeneracy (the true line landing on
the excluded DC bin in BOTH branches at once) would require M1*M2 | q,
i.e. q=0, which is not a valid fundamental. At most one branch can
degenerate for any true alpha0 != 0.
"""
import numpy as np
from twostage import make_signal_exact_N, fine_statistic
from twostage import peak_candidates as _peak_candidates_base


def peak_candidates_with_dc(Tfold, M, gate=3.0, n=4, half=8, guard=2):
    """Like twostage.peak_candidates, but ALSO returns:
      - the mirror (M-p)%M of every found candidate, since a real (rectangular
        or otherwise real-pulse) signal populates +alpha0 and -alpha0 equally
        and the base search only scans the lower half of the spectrum; and
      - p=0 as a separate fallback candidate -- meaningful here because CRT
        can still use a branch's true residue of exactly 0, whereas a
        single-branch search must exclude it as indistinguishable from DC
        leakage."""
    cands = _peak_candidates_base(Tfold, M, gate=gate, n=n, half=half, guard=guard, pmin=1)
    with_mirrors = set(cands) | {(M - p) % M for p in cands}
    with_mirrors.discard(0)
    return sorted(with_mirrors), [0]  # (normal candidates + mirrors, DC fallback)


def verify_candidate(x, lags, alpha_hat, start, B):
    """Direct evaluation of |R_hat^alpha(tau)| at one candidate frequency
    using a short CONTIGUOUS block. This is deliberately independent of
    both decimated branches: checking a candidate against the same samples
    that produced it is circular (the result depends only on q mod M_i,
    which is already known and identical for every combo sharing that
    residue). A contiguous block is not related to either branch by pure
    residue arithmetic, so it discriminates the true alpha from a
    CRT-consistent but physically spurious cross-sign combination. Cost is
    O(B) per lag per candidate (direct sum, not an FFT, since only a
    handful of candidates need checking) -- negligible next to O(M log M)
    as long as the candidate list stays small."""
    idx = start + np.arange(B)
    phase = np.exp(-2j * np.pi * alpha_hat * idx)
    acc = 0.0
    for tau in lags:
        z = x[idx + tau] * np.conj(x[idx])
        acc += np.abs(np.sum(z * phase) / B) ** 2
    return np.sqrt(acc)


def crt_combine(r1, M1, r2, M2):
    """Unique q in [0, M1*M2) with q%M1==r1 and q%M2==r2. Requires gcd(M1,M2)=1."""
    M = M1 * M2
    inv2 = pow(M2, -1, M1)
    inv1 = pow(M1, -1, M2)
    q = (r1 * M2 * inv2 + r2 * M1 * inv1) % M
    return q


def local_floor(x, lags, alpha, B, halfwidth=0.01, n_probes=6):
    """Noise-floor estimate local to `alpha`, not a single global value.

    The non-cyclic (alpha=0) autocorrelation term is always far larger than
    any genuine cyclic feature, and its leakage skirt under a rectangular
    observation window elevates the true local floor near LOW absolute
    frequency well above the floor measured from probes spread uniformly
    over the whole range. Dividing a located peak by successively larger j
    moves the candidate systematically closer to alpha=0 -- exactly into
    that elevated-floor region -- so validating against a single global
    floor lets smaller candidates pass by comparing them to a baseline that
    understates the noise actually present near them.
    """
    lo = max(1e-4, alpha - halfwidth)
    hi = min(0.499, alpha + halfwidth)
    probes = np.linspace(lo, hi, n_probes + 2)[1:-1]
    probes = [p for p in probes if abs(p - alpha) * B > 2]  # skip near-alpha bins
    if not probes:
        probes = [lo, hi]
    return np.median([verify_candidate(x, lags, p, 0, B) for p in probes])


def resolve_harmonic_order(x, lags, N, q_pos, verify_snr, floor, J=12, B=4096):
    """Given the exact bin q_pos of a confidently-located peak, test whether
    q_pos is itself already the fundamental or a harmonic of a SMALLER
    fundamental the peak search never singled out on its own.

    The location step finds the single strongest peak in the record and
    reports its exact bin -- correct as far as it goes, but "strongest
    peak" and "fundamental" only coincide when a comb decays monotonically
    with harmonic order. A broadband comb (e.g. a short-code DSSS signal,
    whose harmonics can be nearly flat across many low orders) has no
    reason to peak at m=1.

    Several fixes were tried before this one:
    - Summed harmonic-consistency score over q_pos/j: fails on a broadband
      comb, because a candidate at a rational fraction of the true
      fundamental still lands on a true comb line at every other one of
      its own harmonics, scoring close enough to win regardless.
    - GCD across every candidate clearing a loose noise-floor gate: fails
      because GCD is fragile to ordinary few-bin position noise.
    - Direct-hit test against a GLOBAL noise floor, with or without a
      second block for confirmation: fails for a different, more basic
      reason -- the near-DC leakage skirt elevates the TRUE local floor at
      low absolute frequency well above the global average (measured:
      ~4x higher for this pulse/rate), and dividing by larger j moves
      candidates systematically toward that low-frequency region, which
      is exactly where the smallest-preferred rule is most exposed to it.
      Nearly every divisor tested cleared a global-floor gate this way,
      regardless of whether it was a real cyclic feature.

    What actually works: gate each candidate against a floor measured
    LOCALLY to that candidate's own frequency (local_floor), which
    correctly reflects the elevated near-DC baseline instead of
    understating it. Among the divisors that still clear their own local
    floor, the smallest is the fundamental, since every other real line is
    by construction an integer multiple of it.

    Cost is O(J*B) for the divisor evaluations plus O(J*n_probes*B) for
    their local floors, independent of N and negligible next to the
    location stage for the small J, n_probes used here.
    """
    if q_pos <= 0:
        return q_pos, [q_pos]
    valid = []
    for j in range(1, J + 1):
        if q_pos % j != 0:
            continue
        q_cand = q_pos // j
        if q_cand == 0:
            continue
        alpha_cand = q_cand / N
        v = verify_candidate(x, lags, alpha_cand, 0, B)
        floor_local = local_floor(x, lags, alpha_cand, B)
        if v >= 6.0 * floor_local:
            valid.append(q_cand)
    if not valid:
        return q_pos, [q_pos]
    return min(valid), sorted(valid)


def coprime_alpha0(x, lags, N, M1, M2, fine_gate=3.0, B_verify=4096,
                    verify_snr=3.0, resolve_order=True, order_J=12):
    """Two-branch coprime disambiguation. Returns (alpha_hat, q, diag).

    verify_snr gates the FINAL accepted combo against a noise-floor estimate
    from the verification block, so a weak/spurious combo is reported as
    "no confident detection" rather than force-picked as the best of a bad
    set -- this matters once more than one cyclic frequency may be present
    (see test_multisignal.py): a real candidate from one branch can otherwise
    pair with the OTHER branch's DC fallback (offered only because that
    branch's true peak failed its own gate, e.g. due to a second signal
    raising the local noise floor) and win purely because nothing better
    was on offer.

    resolve_order applies the divisor/local-floor harmonic-order resolution
    of resolve_harmonic_order (see that function's docstring for why a GCD
    approach was tried and rejected) to every confidently-detected
    candidate, so a broadband/flat comb doesn't get reported at whichever
    harmonic happened to score highest. Disable to inspect the raw located
    peak, e.g. for regression comparison against earlier behavior.

    order_J bounds how far resolve_harmonic_order can descend: it only
    tests divisors j=1..order_J of the located peak, so a true fundamental
    more than order_J times smaller than the peak the location stage
    happens to lock onto is UNREACHABLE -- the function returns the
    located peak's value with no error and no confidence penalty,
    indistinguishable from a genuinely correct detection. This is not
    hypothetical: for a short-code DSSS signal with spreading gain 63
    (chip_rate/data_rate = 63, i.e. requiring order_J>=63 to always be
    safe), the location stage's raw peak landed at 12x the true data rate
    in the majority of trials at 10 dB with the default order_J=12 -- right
    at the boundary, not comfortably inside it -- and in 1/60 trials landed
    at 17x, which order_J=12 cannot resolve, producing a confident but
    wrong final answer. Choose order_J with a safety margin above the
    largest spreading gain / harmonic separation expected in your signals,
    not the default.
    """
    assert N == M1 * M2, "this construction requires N == M1*M2 exactly"
    import math
    assert math.gcd(M1, M2) == 1, "M1, M2 must be coprime"
    D1, D2 = N // M1, N // M2

    T1 = fine_statistic(x, lags, 0, D1, M1)
    T2 = fine_statistic(x, lags, 0, D2, M2)
    c1, dc1 = peak_candidates_with_dc(T1, M1, gate=fine_gate)
    c2, dc2 = peak_candidates_with_dc(T2, M2, gate=fine_gate)
    # DC fallback only when that branch found NOTHING real -- a real
    # candidate is never forced to pair against an artificial 0 when a
    # genuine peer residue is available in the same branch.
    cand1 = list(c1) if c1 else list(dc1)
    cand2 = list(c2) if c2 else list(dc2)

    if not c1 and not c2:
        return None, None, dict(T1=T1, T2=T2, reason="no candidates in either branch")

    combos = []
    for p1 in cand1:
        for p2 in cand2:
            if p1 == 0 and p2 == 0:
                continue
            q = crt_combine(p1, M1, p2, M2)
            combos.append(q)
    combos = sorted(set(combos))
    if not combos:
        return None, None, dict(T1=T1, T2=T2, reason="no valid combination")

    # tie-break the (typically small) candidate list with an INDEPENDENT
    # contiguous block -- not either decimated branch, so it carries real
    # information about which combo is physically correct
    scores = {}
    for q in combos:
        scores[q] = verify_candidate(x, lags, q / N, 0, B_verify)
    best_q = max(scores, key=scores.get)
    best_v = scores[best_q]

    # noise-floor estimate: verify a handful of frequencies unrelated to any
    # combo, same block, same cost class as one more candidate check
    rng_probe = np.random.default_rng(12345)
    probes = rng_probe.uniform(0.001, 0.499, size=8)
    floor = np.median([verify_candidate(x, lags, a, 0, B_verify) for a in probes])

    if best_v < verify_snr * floor:
        return None, None, dict(T1=T1, T2=T2, n_combos=len(combos),
                                 reason="best combo did not clear the noise floor",
                                 best_v=best_v, floor=floor)

    q_pos = min(best_q, N - best_q) if best_q != 0 else 0
    contributing = [q_pos]
    if resolve_order:
        q_pos, contributing = resolve_harmonic_order(x, lags, N, q_pos,
                                                       verify_snr, floor,
                                                       J=order_J, B=B_verify)

    alpha_hat = q_pos / N

    return alpha_hat, q_pos, dict(T1=T1, T2=T2, n_combos=len(combos),
                                   best_v=best_v, floor=floor,
                                   contributing_bins=contributing)


if __name__ == "__main__":
    # sanity: CRT recombination on a clean deterministic bin, no noise
    N = 65280
    M1, M2 = 255, 256
    for q_true in [0, 1, 2560, 33333, 65279]:
        r1, r2 = q_true % M1, q_true % M2
        q_rec = crt_combine(r1, M1, r2, M2)
        assert q_rec == q_true, (q_true, q_rec)
    print("CRT recombination verified on exact residues, all cases correct.")


def adaptive_lags(K, fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """Lag set scaled to the symbol period K = 1/alpha0, in samples.

    A fixed lag set cannot serve a wide range of symbol rates. Two bounds
    constrain every lag tau (see the paper's Section II):

      upper:  tau < K            -- a rectangular pulse of width K has no
                                    cyclic signal beyond its own width, so
                                    such a lag adds only noise variance to
                                    the combined statistic.
      lower:  tau >= gcd(D_i, K) -- branch i samples at stride D_i and so
                                    visits only K/gcd(D_i,K) distinct
                                    within-symbol phases; a shorter lag
                                    never crosses a symbol boundary at any
                                    phase that branch sees, and is dead
                                    regardless of SNR.

    Taking tau as a fixed fraction of K satisfies both automatically, and
    also tracks where the cyclic feature is actually strongest: for a
    rectangular pulse the boundary-crossing probability is ~tau/K, so lags
    at a constant fraction of the symbol period stay in the strong region
    at every rate. Note the bounds are necessary, not sufficient -- a fixed
    set inside them (e.g. {5,6,7}) still fails at large K because tau/K
    becomes tiny.

    Measured over K in 8..200, N=65280, (M1,M2)=(255,256), 15 dB, 20 trials:
    recovery is 0.95-1.00 for BPSK/QPSK/8-PSK at nearly every rate, against
    0.33 at K=10 and 0.08 at K=160-200 for the fixed {4,8,...,24}. The
    exception is 16-QAM at K in {8,16,64}, where gcd(D1,K) >= K leaves
    branch 1 with an empty window and the weakest modulation must be
    carried by branch 2 alone; see test_lag_gcd.py.
    """
    return sorted({max(1, int(round(K * f))) for f in fractions})
