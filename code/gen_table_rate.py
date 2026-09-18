"""
Renders the recovery-rate-vs-symbol-rate sweep (across four modulations)
as a table image, saved as both PDF and PNG. Runs
test_rate_reliability.py's run_rate_sweep() live, so this always
reflects the machine it's run on.

Defaults to the full sps sweep; pass a subset via sps_values= for a
shorter table.
"""
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_rate_reliability import run_rate_sweep, MODS


def render_rate_table(sps_values=None, outfile="table_rate_reliability"):
    kwargs = {} if sps_values is None else dict(sps_values=sps_values)
    results = run_rate_sweep(**kwargs)

    col_labels = ["sps", "rate"] + MODS
    rows = []
    for r in results:
        row = [str(r["sps"]), f"{r['rate']:.4f}"] + [f"{r['recovery'][m]:.2f}" for m in MODS]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(5.5, 0.5 + 0.32 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(f"{outfile}.pdf")
    plt.savefig(f"{outfile}.png", dpi=160)
    print(f"{outfile} written")
    for row in rows:
        print("  " + "  ".join(row))
    return results


if __name__ == "__main__":
    render_rate_table()
