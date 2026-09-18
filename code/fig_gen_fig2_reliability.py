"""
Generates the SNR/two-signal reliability figure (output file:
fig2_reliability.pdf -- the filename is historical and does not match
this script's own name; kept as-is because it is the exact filename
other things reference): (a) recovery rate vs. SNR for a single
signal, (b) outcome fractions with a second, independent
cyclostationary signal present.

The numbers below are copied from the measurements produced by
test_snr_and_bverify.py and test_multisignal.py; this script only
handles the plotting.
"""
import numpy as np
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")  # headless-safe default for standalone `python3` execution
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.3))

# (a) SNR floor, single signal -- plotted low-to-high, left to right
snr = [-2, 0, 2, 4, 6, 10]
rec_snr = [0.00, 0.00, 0.25, 0.80, 0.98, 0.98]
ax[0].plot(snr, rec_snr, "-o", ms=4, color="#4C72B0")
ax[0].set_xlabel("SNR (dB)", fontsize=8)
ax[0].set_ylabel("recovery rate", fontsize=8)
ax[0].tick_params(labelsize=7)
ax[0].grid(alpha=0.3)
ax[0].set_title("(a)", fontsize=8)

# (b) multi-signal robustness after the harmonic-order-resolution fix
ratios = ["1.0", "0.3", "0.1"]
hit  = [29/60, 60/60, 60/60]
spur = [8/60, 0/60, 0/60]
none = [23/60, 0/60, 0/60]
x = np.arange(3); w = 0.25
ax[1].bar(x-w, hit, w, label="correct", color="#55A868")
ax[1].bar(x, spur, w, label="spurious", color="#C44E52")
ax[1].bar(x+w, none, w, label="no detect.", color="#8C8C8C")
ax[1].set_xticks(x); ax[1].set_xticklabels(ratios, fontsize=7)
ax[1].set_xlabel("power ratio $b/a$", fontsize=8)
ax[1].legend(fontsize=6, frameon=False)
ax[1].tick_params(labelsize=7)
ax[1].grid(alpha=0.3, axis="y")
ax[1].set_title("(b)", fontsize=8)

plt.tight_layout()
plt.savefig("fig2_reliability.pdf")
plt.savefig("fig2_reliability.png", dpi=160)
print("fig2_reliability written")
