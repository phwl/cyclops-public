"""
Renders the S3CA comparison (timing + accuracy) as a table image, saved
as both PDF and PNG. Runs compare_s3ca.py's run_s3ca_comparison() live,
so this always reflects the machine it's run on.
"""
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_s3ca import run_s3ca_comparison

results = run_s3ca_comparison()

col_labels = ["N", "S3CA dense (ms)", "S3CA sfft1 (ms)", "coprime (ms)", "speedup"]
rows = []
for r in results:
    rows.append([
        f"{r['N']:,}",
        f"{r['s3ca_dense_ms']:.1f}",
        f"{r['s3ca_sfft1_ms']:.1f}",
        f"{r['coprime_ms']:.1f}",
        f"{r['speedup']:.1f}x",
    ])

fig, ax = plt.subplots(figsize=(5.5, 0.5 + 0.35 * len(rows)))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig("table_s3ca_comparison.pdf")
plt.savefig("table_s3ca_comparison.png", dpi=160)
print("table_s3ca_comparison written")
for row in rows:
    print("  " + "  ".join(row))
