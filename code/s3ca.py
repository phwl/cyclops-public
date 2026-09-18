"""
S3CA -- Sparse Strip Spectral Correlation Analyzer
====================================================

Implementation of

    C. J. Li, R. Rademacher, D. Boland, C. T. Jin, C. M. Spooner, P. H.W. Leong,
    "S3CA: A Sparse Strip Spectral Correlation Analyzer", IEEE SPL 2015-style
    letter (as supplied, s3ca_spl24.pdf).

built on top of the sfft_opt.py sparse-FFT backend (sFFT 1.0, random
hashing; Hassanieh, Indyk, Katabi, Price, SODA'12), used strictly as a
library (unmodified).

This is a deliberately narrowed version of the original implementation,
which also supported a fixed-decimation backend, an RFFAST backend, a
"naive" S3CA mode, and a multi-backend check()/compare_backends()
verification harness. None of that is used anywhere this file is
imported from -- only dense_ssca() and s3ca()'s COMPIDX ("full") mode on
the sfft1 backend are ever called -- so all of it was removed rather
than carried along unused. See git history / the original upload if any
of that is needed again.

------------------------------------------------------------------------------
What this file provides
------------------------------------------------------------------------------
dense_ssca(x, Np, ...)
    The conventional strip spectral correlation analyzer (SSCA), computed
    densely with ordinary FFTs. This is the ground-truth reference S3CA is
    checked against.

s3ca(x, Np, kappa, ...)
    The sparse strip spectral correlation analyzer: COMPIDX. The same
    sigma/tau (Algorithm 2's Sigma, Upsilon) are used for every channel's
    SFFT, so the set of time-domain samples any channel's SFFT will ever
    look at is *identical* across channels. That shared set W' is computed
    once, and the channelizer + channel-data-product (CDP) -- the
    O(N*Np*log Np) part of the SSCA -- is evaluated only at the indices in
    W' (dilated by the channelizer's own Np-sample window -- see
    `_channelizer_footprint` -- since computing X_T at time t needs a
    small window of raw samples around t, not just x[t]), not over the
    full length-N signal.

------------------------------------------------------------------------------
Modeling choices / simplifications (stated explicitly)
------------------------------------------------------------------------------
* Circular convention. The paper's Eq. (1)-(2) need N + Np raw samples (Np/2
  of "halo" on each side of an N-sample block) to form X_T and X_g without
  edge effects. To keep this a clean, self-contained, single-block demo we
  treat the length-N input as one period of a periodic signal, i.e. the
  channelizer wraps circularly. Ts = 1/fs = 1 (as in the paper's
  normalization), so f_k = k/Np exactly and Delta_alpha = 1/N exactly.
  This changes nothing about the SFFT-acceleration logic (the whole point of
  the exercise); it only avoids a boundary-sample bookkeeping detail.

* Windows a(r) (channelizer taper) and g(m) (outer window) are Hamming by
  default and are shared between dense_ssca and s3ca, so comparisons between
  them are apples-to-apples regardless of which window you pick.

* What is and isn't saved, precisely. The paper claims two savings: (1) the
  channelizer/CDP need only be evaluated at |W'| time positions instead of N
  (Table I row 1: O(N*Np*log Np) -> O(|W'|*Np*log Np) -- this is a count of
  Np-point FFTs run, and is genuinely realized here: s3ca() runs |W'| of
  them, not N), and (2) the intermediate CDP matrix X'_g is |W'| x Np, not
  N x Np (paper's second bullet). Both of those are realized in
  `_channelizer_rows`/below: they operate on |W'|-length arrays. What is
  *not* fully realized: `sfft1`'s public API takes one dense length-N
  array, so each per-channel CDP column still gets scattered into an
  N-length (mostly zero) buffer before being handed to `sfft1` -- an O(N)
  allocation per channel, even though only |W'| of it is ever read.
  Avoiding that would mean restructuring sfft1's internals to accept a
  sparse/dict input, which is out of scope for using the library as-is, so
  it is left as a documented gap.

  Separately: the *distinct raw x[] samples* the restricted channelizer
  reads (each of the |W'| positions needs a small Np-sample window, not
  just one sample) is a different, larger quantity that can approach N even
  when |W'| is small -- see `_channelizer_footprint`. That is not one of
  the paper's two claimed savings and is reported separately (as
  `n_raw_samples_read` on the returned result), for transparency, rather
  than folded into the headline sparsity number.
"""

from __future__ import annotations

import time
from math import gcd
from typing import NamedTuple

import numpy as np

from sfft_opt import Filter, flat_filter, sfft1

__all__ = ["dense_ssca", "s3ca", "S3CAResult"]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _centered(m):
    """Index range [-m//2, m - m//2) -- matches the paper's k in [-Np/2,Np/2-1)
    and q in [-N/2, N/2-1) conventions for even m."""
    return np.arange(-(m // 2), m - m // 2)


def _default_windows(N, Np):
    return np.hamming(Np), np.hamming(N)


def _to_f_alpha(k_centered, q_centered, Np, N):
    """Eq. (3)'s coordinate map, normalised to fs = 1."""
    fk = k_centered / Np
    dalpha = 1.0 / N
    alpha = fk + q_centered * dalpha
    f = (fk - q_centered * dalpha) / 2.0
    return f, alpha


# ---------------------------------------------------------------------------
# dense channelizer + CDP  (shared machinery, full-array version)
# ---------------------------------------------------------------------------
def _channelizer_dense(x, Np, a):
    """X_T(t,k) for every t in [0,N) and every channel k (centered), via one
    batched Np-point FFT per t.  O(N*Np*log Np) -- see Table I, SSCA row 1."""
    N = x.size
    k_idx = _centered(Np)
    r_off = np.arange(-(Np // 2), Np - Np // 2)             # matches k_idx pattern

    # S_mat[t, r'] = a(r) * x[(t + r) mod N],  r' indexes r_off in order
    t = np.arange(N)
    S_mat = np.empty((N, Np), dtype=complex)
    for ri, r in enumerate(r_off):
        S_mat[:, ri] = a[ri] * x[(t + r) % N]

    bracket = np.fft.fft(S_mat, axis=1)                     # fft order k' = 0..Np-1
    kprime = np.arange(Np)
    bracket *= np.exp(1j * np.pi * kprime)[None, :]         # undo r-offset phase
    order = k_idx % Np
    bracket = bracket[:, order]                              # reorder -> k_idx order

    fk = k_idx / Np
    XT = bracket * np.exp(-2j * np.pi * fk[None, :] * t[:, None])
    return XT, k_idx


def _channelizer_footprint(t_idx, N, Np):
    """The raw-signal samples the channelizer actually reads to evaluate
    X_T at every t in `t_idx`: each t needs a Np-sample window x[t-Np/2 :
    t+Np/2) (circularly), not just x[t] itself. This dilated set, not
    `t_idx` alone, is the true number of samples the restricted
    channelizer touches."""
    r_off = np.arange(-(Np // 2), Np - Np // 2)
    footprint = (np.asarray(t_idx)[:, None] + r_off[None, :]) % N
    return np.unique(footprint.ravel())


def _channelizer_rows(x, Np, a, t_idx):
    """Same as `_channelizer_dense` but only at the rows in `t_idx` --
    O(|t_idx| * Np * log Np).  This is the S3CA channelizer restriction."""
    N = x.size
    k_idx = _centered(Np)
    r_off = np.arange(-(Np // 2), Np - Np // 2)

    t_idx = np.asarray(t_idx)
    S_mat = np.empty((t_idx.size, Np), dtype=complex)
    for ri, r in enumerate(r_off):
        S_mat[:, ri] = a[ri] * x[(t_idx + r) % N]

    bracket = np.fft.fft(S_mat, axis=1)
    kprime = np.arange(Np)
    bracket *= np.exp(1j * np.pi * kprime)[None, :]
    order = k_idx % Np
    bracket = bracket[:, order]

    fk = k_idx / Np
    XT = bracket * np.exp(-2j * np.pi * fk[None, :] * t_idx[:, None])
    return XT, k_idx


# ---------------------------------------------------------------------------
# dense SSCA -- the reference / ground truth
# ---------------------------------------------------------------------------
def dense_ssca(x, Np, a=None, g=None):
    """Conventional (dense) strip spectral correlation analyzer.

    Parameters
    ----------
    x : complex array, length N.
    Np : number of channelizer bands (must divide... only needs to be even).
    a : length-Np channelizer taper (default Hamming).
    g : length-N outer window (default Hamming).

    Returns
    -------
    S : complex array, shape (N, Np).  S[qi, ki] is the SCD estimate at
        (f, alpha) = f_alpha[qi, ki, 0], f_alpha[qi, ki, 1].
    f, alpha : real arrays, shape (N, Np), the coordinate grids.
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g

    XT, k_idx = _channelizer_dense(x, Np, a)
    Xg = XT * np.conj(x)[:, None] * g[:, None]

    Sfft = np.fft.fft(Xg, axis=0)                # bin q' = 0..N-1
    q_idx = _centered(N)
    S = Sfft[q_idx % N, :]

    f, alpha = _to_f_alpha(k_idx[None, :], q_idx[:, None], Np, N)
    return S, f, alpha


# ---------------------------------------------------------------------------
# COMPIDX: which time samples will the shared-seed SFFTs touch?
# ---------------------------------------------------------------------------
def required_time_indices(N, filt: Filter, loc_loops, est_loops, seed):
    """Reproduces exactly the sigma/tau draw and index-gather that `sfft1`
    performs internally (sfft_opt.py, the "sigmas/taus" and "idx" computation
    right before the batched FFT loop), so that calling this with the same
    (N, filt, loc_loops, est_loops, seed) that every per-channel `sfft1` call
    below will use tells us, in advance, the union of samples any of those
    calls could ever read -- without running any of them.

    This is COMPIDX (Algorithm 2) realized by reusing sfft1's own public
    seeding contract rather than reimplementing sfft1.
    """
    L = loc_loops + est_loops
    rng = np.random.default_rng(seed)
    sigmas = np.empty(L, dtype=np.int64)
    taus = np.empty(L, dtype=np.int64)
    for i in range(L):
        s = int(rng.integers(0, N))
        while gcd(s, N) != 1:
            s = int(rng.integers(0, N))
        sigmas[i] = s
        taus[i] = int(rng.integers(0, N))

    supp = np.arange(filt.Wp, dtype=np.int64)
    idx = (sigmas[:, None] * supp[None, :] + taus[:, None]) % N
    return np.unique(idx.ravel()), sigmas, taus


def _auto_filter(N, kappa, B=None, tolerance=1e-6, box_scale=1.6):
    """sfft1 needs a Filter sized to N and the target sparsity kappa; this
    is the same B-selection heuristic the original multi-backend version
    used for its sfft1 backend."""
    if B is None:
        B = 1 << max(1, int(round(0.5 * np.log2(max(N * kappa / 5.0, 4)))))
        B = min(B, N)
        while N % B:
            B //= 2
    return flat_filter(N, B, tolerance=tolerance, box_scale=box_scale)


# ---------------------------------------------------------------------------
# S3CA
# ---------------------------------------------------------------------------
class S3CAResult(NamedTuple):
    f: np.ndarray            # spectral frequency, one per recovered (channel, freq) pair
    alpha: np.ndarray        # cycle frequency
    value: np.ndarray        # complex SCD estimate
    channel: np.ndarray      # which channel k (centered index) each entry came from
    n_positions: int         # |W'|: how many time positions the channelizer/CDP were
                              # evaluated at, i.e. how many Np-point FFTs were run. This
                              # is the quantity that drives the paper's Table I compute
                              # and (Np-column) intermediate-storage savings: |W'| << N.
    n_raw_samples_read: int  # distinct raw x[] samples actually read to do that -- >=
                              # n_positions, since each position needs a small Np-sample
                              # window around it. Not one of the paper's two claimed
                              # savings dimensions, and can approach N even when
                              # n_positions is small if W' is dense enough that the
                              # Np-windows tile over it; reported for transparency.
    elapsed: float           # wall time in seconds


def s3ca(x, Np, kappa, seed=0, a=None, g=None,
         filt=None, B=None, loc_loops=4, est_loops=16,
         tolerance=1e-6, box_scale=1.6) -> S3CAResult:
    """Sparse strip spectral correlation analyzer (COMPIDX, sfft1 backend).

    Parameters
    ----------
    x : complex array, length N.
    Np : number of channelizer bands.
    kappa : target number of non-zero cyclic-spectrum coefficients per
        channel.
    seed : rng seed for sfft1's random hashing. Reused, unchanged, for
        every channel's sparse-recovery call -- that reuse is exactly what
        makes the required-sample set identical across channels (see
        module docstring and `required_time_indices`), which is what lets
        the channelizer/CDP be restricted to |W'| positions in the first
        place.
    filt, B, loc_loops, est_loops, tolerance, box_scale : sfft1 tuning
        parameters. If `filt` is not given, one is built automatically
        from `N`/`kappa` (see `_auto_filter`); `B` overrides just the
        bucket count within that auto-construction.

    Returns
    -------
    S3CAResult
    """
    x = np.asarray(x)
    N = x.size
    if a is None or g is None:
        a_def, g_def = _default_windows(N, Np)
        a = a_def if a is None else a
        g = g_def if g is None else g
    if filt is None:
        filt = _auto_filter(N, kappa, B=B, tolerance=tolerance, box_scale=box_scale)

    k_idx = _centered(Np)
    t0 = time.perf_counter()

    # COMPIDX: the required-sample set is identical across channels (the
    # same-seed trick -- see required_time_indices). Compute it once, then
    # only fill in the channelizer + CDP at those samples.
    Wp_idx, _, _ = required_time_indices(N, filt, loc_loops, est_loops, seed)
    n_positions = Wp_idx.size
    n_raw = _channelizer_footprint(Wp_idx, N, Np).size

    XT_rows, _ = _channelizer_rows(x, Np, a, Wp_idx)
    Xg_rows = XT_rows * np.conj(x[Wp_idx])[:, None] * g[Wp_idx][:, None]

    fs, alphas, values, chans = [], [], [], []
    for ci, k in enumerate(k_idx):
        Xg_col = np.zeros(N, dtype=complex)
        Xg_col[Wp_idx] = Xg_rows[:, ci]
        freqs, coeffs = sfft1(Xg_col, kappa, filt=filt, loc_loops=loc_loops,
                               est_loops=est_loops, rng=seed)
        q_centered = np.where(freqs < N // 2, freqs, freqs - N)
        f, alpha = _to_f_alpha(k, q_centered, Np, N)
        fs.append(f); alphas.append(alpha); values.append(coeffs)
        chans.append(np.full(freqs.size, k))

    elapsed = time.perf_counter() - t0
    f = np.concatenate(fs) if fs else np.empty(0)
    alpha = np.concatenate(alphas) if alphas else np.empty(0)
    value = np.concatenate(values) if values else np.empty(0, dtype=complex)
    channel = np.concatenate(chans) if chans else np.empty(0, dtype=int)
    return S3CAResult(f, alpha, value, channel, n_positions, n_raw, elapsed)
