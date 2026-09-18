# Reproducing the experiments

Open `reproduce_paper.ipynb` and run all cells, from this directory. It runs every
experiment in this codebase and generates a PDF (and PNG) for every figure and table,
by calling the same scripts below -- nothing in the notebook is a separately re-derived
version of any result.

Everything lives flat in this one directory on purpose: every script's imports resolve
against its own directory automatically, so nothing here needs `PYTHONPATH` or any
`sys.path` setup, whether run from the notebook or directly (`python3 test_dsss.py`,
`python3 compare_s3ca.py`, etc.).

Most cells finish in seconds. Two are slow: the S3CA comparison (well under a minute
here) and the full rate/modulation sweep (sps 8-200 x 4 modulations x 20 trials).

Wall-clock numbers (timing figures/tables) are inherently machine- and load-dependent;
re-run on an otherwise-idle machine for a clean measurement, and increase the `reps`
argument in `test_cost_and_profile.py` / `fig_gen_fig1_cost.py` for less run-to-run
jitter.

## Files

- `coprime.py` — the coprime dual-rate estimator: folding, CRT recombination,
  sign-ambiguity handling, verification, harmonic-order resolution, profile readout,
  degenerate-branch handling.
- `twostage.py` — the earlier single-decimation estimator, kept because
  `test_coprime_correctness.py` compares against it directly (it's the "a single
  decimation fails outright" baseline) and because `coprime.py` reuses its
  `fine_statistic` / `peak_candidates` helpers.
- `fold_core.py` — the folding identity and its float-precision check.
- `signal_model.py` — the linearly-modulated test-signal generator (rectangular or RRC
  chip shaping, any of BPSK/QPSK/8PSK/16QAM, additive noise, carrier offset).
- `offgrid_gen.py` — signal generator for a symbol rate that does not land on an
  integer bin (real-valued samples-per-symbol via rounded symbol boundaries).
- `test_sparse_storage.py` — verifies the storage claim: the pipeline reads only
  O(sqrt(N) + B) of the N sample positions, since every position it touches is
  fixed in advance by (M1, M2, lags, B). Checked both by counting positions and
  by overwriting all others with NaN and confirming recovery is unchanged.
- `test_adaptive_lags.py` — compares the fixed lag set against `adaptive_lags(K)`
  from `coprime.py`, which scales the lags with the symbol period so both bounds
  are satisfied at every rate; raises mean recovery over K=8..200 from 0.69 to
  0.95 across four modulations.
- `test_lag_gcd.py` — characterizes the lag-set structural null: a lag tau
  contributes zero cyclic signal whenever tau < gcd(D, sps) for a branch's
  decimation stride D, which can silence the entire fixed lag set on one
  branch for symbol periods sharing large factors with that stride.
- `test_*.py` — one script per experiment; most also expose a `run_*()` function
  (e.g. `run_cost_sweep()`, `run_s3ca_comparison()`, `run_rate_sweep()`) that returns
  the same numbers as structured data, for use by the `gen_*`/`fig_gen_*` scripts and
  the notebook, instead of only printing to stdout.
- `fig_gen_fig1_cost.py`, `fig_gen_fig2_dsss.py`, `fig_gen_fig2_reliability.py` —
  figure generation, each producing both a `.pdf` and a `.png`. The *output* filenames
  (`fig1_cost.pdf`, `fig3_dsss_profile.pdf`, `fig2_reliability.pdf`) don't match the
  script names 1:1 for historical reasons documented in each script's own docstring;
  treat the docstring, not the filename, as authoritative for what each script does.
- `gen_table_s3ca.py`, `gen_table_rate.py` — table generation (S3CA comparison,
  recovery-rate-vs-symbol-rate sweep), each producing both a `.pdf` and a `.png` via a
  rendered matplotlib table, computed live from the corresponding `test_*`/`compare_*`
  script's `run_*()` function.
- `s3ca.py`, `sfft_opt.py` — the S3CA implementation used for comparison: `dense_ssca`
  (the dense ground-truth reference) and `s3ca` (the sparse recovery, built on the
  `sfft1` random-hashing sparse FFT in `sfft_opt.py`). This is a narrowed rewrite of a
  larger implementation that was supplied for this comparison -- see "What was
  simplified" below.
- `compare_s3ca.py` — the timing/accuracy comparison against S3CA. At
  $N=4{,}192{,}256$, `dense_ssca` needs several GB for its intermediate arrays; that
  record length is omitted from the S3CA comparison for this reason (also why the S3CA
  curve in the cost figure stops earlier than the other two).

## What was simplified in s3ca.py

The version originally supplied also included a fixed-decimation sparse-FFT backend, an
RFFAST backend, a "naive" S3CA mode (dense channelizer with independent per-channel
recovery), and a multi-backend `check()`/`compare_backends()` verification harness for
comparing backends against each other. None of that is exercised anywhere in this
codebase -- only `dense_ssca` and `s3ca`'s COMPIDX ("full") mode on the `sfft1` backend
are ever called, so everything else was removed rather than carried along unused:

- `decimated_sfft.py` and `rffast.py` are gone (nothing imports them anymore).
- `s3ca()` lost its `mode` and `backend` parameters -- there's only one mode and one
  backend now, so the sfft1 tuning parameters that used to be passed through a generic
  `**backend_kwargs` (`filt`, `B`, `loc_loops`, `est_loops`, `tolerance`, `box_scale`)
  are now direct, named parameters instead.
- `check()`, `compare_backends()`, `CheckReport`, `print_report()`, `print_comparison()`,
  and the `_SFFT1Backend`/`_DecimatedBackend` abstraction layer are all gone; `s3ca()`
  calls `sfft1()` directly rather than through a pluggable-backend interface that only
  ever had one thing plugged into it.

Verified after rewriting, not just assumed: reran the full comparison and confirmed
every profile value, at every harmonic and every record length, matches the
pre-simplification numbers exactly (e.g. the dense/sfft1 pair at $N=65{,}280$, $m=1$:
86711.1 / 89006.7, unchanged to the decimal) -- the rewrite removed unused code paths
without touching the arithmetic of the ones actually used.

## What was removed from the original working set (earlier cleanup)

`rrc_check.py`, `rrc_sweep.py`, `coprime_sweeps_fig.py`, `twostage_summary_fig.py`, and
`cost_compare.py` were exploratory scripts from an earlier, unrelated pulse-shaping
project and superseded intermediate versions of the cost and DSSS figures. Nothing here
depends on them. The one genuine dependency among them, `rrc_check.py`'s signal
generator, was extracted into `signal_model.py` rather than removed outright.

An earlier version of this directory also split things into `ours/` and `s3ca_library/`
subdirectories, which required setting `PYTHONPATH` for `compare_s3ca.py` to find both.
Flattened back to one directory instead: every filename here is unique, nothing was
gained by the split, and it's one less thing to explain or get wrong when running this
elsewhere.
