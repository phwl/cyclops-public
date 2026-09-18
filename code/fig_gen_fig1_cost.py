"""
Generates the cost-vs-record-length figure (output file: fig1_cost.pdf):
measured wall-clock cost vs. record length N for a single full-length
FFT evaluation, S3CA (sfft1 backend), and the coprime estimator's
combined location + profile-readout cost.

Runs the underlying measurements live (via test_cost_and_profile.py's
run_cost_sweep() and compare_s3ca.py's run_s3ca_comparison()) rather than
using pasted-in numbers, so this always reflects the machine it's run on.
The S3CA curve stops earlier than the other two because dense_ssca's
memory footprint becomes impractical beyond N~1e6 (see compare_s3ca.py).

Wall-clock timings are inherently noisy on a shared/virtual machine; for
a final, publication-quality run, increase `reps` below and run on an
otherwise-idle machine.
"""
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")  # headless-safe default for standalone `python3` execution
import matplotlib.pyplot as plt

from test_cost_and_profile import run_cost_sweep
from compare_s3ca import run_s3ca_comparison

REPS = 7  # bump to 10-20 for a final run; see test_cost_and_profile.py

cost_results = run_cost_sweep(reps=REPS)
N = [r["N"] for r in cost_results]
t_base = [r["baseline_ms"] for r in cost_results]
t_total = [r["total_ms"] for r in cost_results]

s3ca_results = run_s3ca_comparison()  # stops at N~1e6, see docstring above
N_s3ca = [r["N"] for r in s3ca_results]
t_s3ca = [r["s3ca_sfft1_ms"] for r in s3ca_results]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(N, t_base, "-o", ms=4, label="single full-length FFT", color="#4C72B0")
ax.plot(N_s3ca, t_s3ca, "-^", ms=4, label="S3CA (sfft1)", color="#55A868")
ax.plot(N, t_total, "-s", ms=4, label="CYCLOPS (location+profile)", color="#C44E52")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("record length $N$")
ax.set_ylabel("wall-clock time (ms)")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("fig1_cost.pdf")
plt.savefig("fig1_cost.png", dpi=160)
print("fig1_cost written")
for r in cost_results:
    print(f"  N={r['N']:>9}  baseline={r['baseline_ms']:8.2f} ms  "
          f"total={r['total_ms']:7.2f} ms  speedup={r['speedup']:.1f}x")
