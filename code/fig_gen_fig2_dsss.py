"""
Generates the DSSS alpha-profile figure (output file: fig3_dsss_profile.pdf
-- the filename is historical and does not match this script's own name;
it is kept as-is because it is the exact filename other things reference).
Shows both the full dense spectrum at every native bin (thin gray line)
and, at each harmonic order m of +-alpha0, the dense reference (open
circles) against the coprime estimator's own readout (filled dots,
standard B=4096 block), averaged over 20 trials. The x-axis is cycle
frequency expressed as a continuous multiple of alpha0, so the full
spectrum and the discrete per-harmonic points share one axis.

This also reports recovery rate over the 20 trials, and the mean
relative profile error across all 18 harmonics.

Run from the same directory as coprime.py, twostage.py, and
test_dsss.py (for the DSSS signal generator).
"""
import numpy as np
import matplotlib
try:
    get_ipython()  # defined only inside IPython/Jupyter
except NameError:
    matplotlib.use("Agg")  # headless-safe default for standalone `python3` execution
import matplotlib.pyplot as plt
from coprime import coprime_alpha0, verify_candidate
from test_dsss import make_dsss_signal

N = 61752                  # = 248 * 249, coprime, both multiples compatible
M1, M2 = 248, 249           # with the true symbol period (124 samples)
lags = [4, 8, 12, 16, 20, 24]
chip_rate, spreading_gain = 0.25, 31
a0 = 1.0 / (round(1 / chip_rate) * spreading_gain)   # = 1/124
SNR = 10.0
H = 9                       # harmonics m=1..9 on each side of +-alpha0
Bprof = 4096
NTRIALS = 20


def profile_at_harmonics(x, q0, B, H, sign=+1):
    """P(m*alpha0) for m=1..H (sign=+1) or m=-1..-H (sign=-1), via V(alpha)."""
    out = np.zeros(H)
    for i, m in enumerate(range(1, H + 1)):
        q = (sign * m * q0) % N
        out[i] = verify_candidate(x, lags, q / N, 0, B)
    return out


if __name__ == "__main__":
    Pfull_pos = np.zeros((NTRIALS, H))
    Pfull_neg = np.zeros((NTRIALS, H))
    Phat_pos = np.zeros((NTRIALS, H))
    Phat_neg = np.zeros((NTRIALS, H))
    dense_curve = None
    q0_first = None
    n_correct = 0

    for tr in range(NTRIALS):
        rng = np.random.default_rng(9000 + tr)
        x, _, _ = make_dsss_signal(chip_rate, spreading_gain, N, SNR, rng)
        ah, q0, diag = coprime_alpha0(x, lags, N, M1, M2)
        n_correct += ah is not None and abs(ah - a0) < 0.5 / N

        if tr == 0:
            # full dense spectrum, one length-N FFT per lag, at every
            # native bin -- the background trace showing the noise floor
            # the harmonics sit above, not just the harmonics themselves
            acc = np.zeros(N)
            for tau in lags:
                z = x[np.arange(N) + tau] * np.conj(x[np.arange(N)])
                acc += np.abs(np.fft.fft(z) / N) ** 2
            dense_curve = np.sqrt(acc)
            q0_first = q0

        Pfull_pos[tr] = profile_at_harmonics(x, q0, N, H, sign=+1)
        Pfull_neg[tr] = profile_at_harmonics(x, q0, N, H, sign=-1)
        Phat_pos[tr] = profile_at_harmonics(x, q0, Bprof, H, sign=+1)
        Phat_neg[tr] = profile_at_harmonics(x, q0, Bprof, H, sign=-1)

    print(f"fundamental recovery: {n_correct}/{NTRIALS}")

    Pfull_pos_m, Pfull_neg_m = Pfull_pos.mean(0), Pfull_neg.mean(0)
    Phat_pos_m, Phat_neg_m = Phat_pos.mean(0), Phat_neg.mean(0)
    rel = np.concatenate([
        np.abs(Phat_pos_m - Pfull_pos_m) / Pfull_pos_m,
        np.abs(Phat_neg_m - Pfull_neg_m) / Pfull_neg_m,
    ])
    print(f"mean relative profile error across all {2*H} harmonics: {rel.mean():.3f}")

    harm_m = np.concatenate([-np.arange(H, 0, -1), np.arange(1, H + 1)])
    harm_dense = np.concatenate([Pfull_neg_m[::-1], Pfull_pos_m])
    harm_est = np.concatenate([Phat_neg_m[::-1], Phat_pos_m])

    # full dense spectrum, x-axis rescaled to continuous multiples of
    # alpha0 (q/q0) so it shares an axis with the discrete harmonic points
    qmax = int(round((H + 0.5) * q0_first))
    q_display = np.arange(-qmax, qmax + 1)
    x_display = q_display / q0_first
    dense_display = dense_curve[q_display % N]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(x_display, dense_display, color="0.75", lw=0.5, label="dense (all $\\alpha$)")
    ax.plot(harm_m, harm_dense, "o", ms=7, mfc="none", mec="0.15", mew=1.1,
            label="dense (harmonics)")
    ax.plot(harm_m, harm_est, "o", ms=4.2, color="#E37B22", label="coprime estimate")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 8)   # headroom so the legend clears the harmonic row
    ax.set_xticks(harm_m)
    ax.set_xticklabels([str(m) for m in harm_m], fontsize=6)
    ax.set_xlabel(r"cycle frequency (multiples of $\alpha_0$)", fontsize=9)
    ax.set_ylabel(r"$P(\alpha)$", fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=6, frameon=True, loc="upper center", ncol=1)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("fig3_dsss_profile.pdf")
    plt.savefig("fig3_dsss_profile.png", dpi=160)
    print("fig3_dsss_profile written")
