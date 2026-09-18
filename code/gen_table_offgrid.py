"""
Renders the off-grid recovery-outcome sweep as a table image, saved as
both PDF and PNG. Runs test_offgrid.py's run_offgrid_sweep() live.
"""
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_offgrid import run_offgrid_sweep

results = run_offgrid_sweep()

col_labels = ["delta (bins)", "exact", "wild", "no-detection"]
rows = [[f"{r['delta']:.2f}", f"{r['exact']:.2f}", f"{r['wild']:.2f}", f"{r['no_detect']:.2f}"]
        for r in results]

fig, ax = plt.subplots(figsize=(4.5, 0.5 + 0.35 * len(rows)))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig("table_offgrid.pdf")
plt.savefig("table_offgrid.png", dpi=160)
print("table_offgrid written")
for row in rows:
    print("  " + "  ".join(row))
