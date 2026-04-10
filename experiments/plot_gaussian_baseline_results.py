import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run_gaussian_slam_baseline import run

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
seeds = [0, 1, 2, 3, 4]
TAU = 3.96

# ------------------------------------------------------------------ #
# Collect results                                                      #
# ------------------------------------------------------------------ #

rmse_means, rmse_stds = [], []
nees_means, nees_stds = [], []
prec_means, prec_stds = [], []
rec_means, rec_stds = [], []
trigger_means = []
true_failure_means = []
scores_all = []

for noise in noise_levels:
    rmses, neess, precs, recs, trigs, true_fails, scs = [], [], [], [], [], [], []

    for seed in seeds:
        r = run(assoc_noise=noise, tau=TAU, seed=seed, verbose=False)
        rmses.append(r["rmse"])
        neess.append(r["nees"])
        precs.append(
            float(r["precision"])
            if r["precision"] is not None and not np.isnan(float(r["precision"]))
            else float("nan")
        )
        recs.append(
            float(r["recall"])
            if r["recall"] is not None and not np.isnan(float(r["recall"]))
            else float("nan")
        )
        trigs.append(len(r["triggers"]))
        true_fails.append(len(r["true_failures"]))
        scs.extend([s for _, s in r["scores"]])

    rmse_means.append(np.mean(rmses))
    rmse_stds.append(np.std(rmses))
    nees_means.append(np.mean(neess))
    nees_stds.append(np.std(neess))
    prec_means.append(np.nanmean(precs))
    prec_stds.append(np.nanstd(precs))
    rec_means.append(np.nanmean(recs))
    rec_stds.append(np.nanstd(recs))
    trigger_means.append(np.mean(trigs))
    true_failure_means.append(np.mean(true_fails))
    scores_all.append(scs)

    print(
        f"noise={noise:.1f} | "
        f"RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f} | "
        f"NEES={np.mean(neess):.1f}±{np.std(neess):.1f} | "
        f"P={np.nanmean(precs):.2f} R={np.nanmean(recs):.2f} | "
        f"triggers={np.mean(trigs):.1f}/28 | "
        f"true_failures={np.mean(true_fails):.1f}/28"
    )

# ------------------------------------------------------------------ #
# Plot                                                                 #
# ------------------------------------------------------------------ #

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Baseline: iSAM2 + Condition-Number Trigger + Rejection Sampling Refinement\n"
    f"5 seeds × {len(noise_levels)} noise levels, τ = {TAU}",
    fontsize=13,
)

kw_eb = dict(fmt="o-", linewidth=2, capsize=5, capthick=1.5, markersize=6)

# ── (0,0) Pose RMSE ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.errorbar(
    noise_levels,
    rmse_means,
    yerr=rmse_stds,
    color="steelblue",
    label="mean ± std",
    **kw_eb,
)
ax.axhline(0.5, color="gray", ls="--", alpha=0.6, label="0.5 m reference")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Pose RMSE (m)")
ax.set_title(
    "Robot pose error vs data-association noise\n"
    "iSAM2 MAP estimate, no non-Gaussian correction"
)
ax.legend()
ax.grid(True, alpha=0.3)

# ── (0,1) NEES ───────────────────────────────────────────────────────
ax = axes[0, 1]
ax.errorbar(
    noise_levels, nees_means, yerr=nees_stds, color="coral", label="mean ± std", **kw_eb
)
ax.axhline(2.0, color="green", ls="--", lw=1.5, label="Consistent filter (NEES = 2)")
ax.set_yscale("log")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Mean NEES (log scale)")
ax.set_title(
    "Filter consistency vs data-association noise\nNEES ≫ 2 → overconfident covariance"
)
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# ── (0,2) Trigger count vs true failures ─────────────────────────────
ax = axes[0, 2]
x = np.array(noise_levels)
w = 0.03
ax.bar(x - w, trigger_means, width=w * 2, color="steelblue", label="Windows triggered")
ax.bar(
    x + w, true_failure_means, width=w * 2, color="salmon", label="True failure windows"
)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Mean window count (out of 28)")
ax.set_title(
    f"Trigger count vs actual failures (τ = {TAU})\n"
    "Gap reveals missed detections and false alarms"
)
ax.set_ylim(0, 30)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for xi, t, f in zip(x, trigger_means, true_failure_means):
    if t > 0:
        ax.text(xi - w, t + 0.4, f"{t:.0f}", ha="center", fontsize=8)
    if f > 0:
        ax.text(xi + w, f + 0.4, f"{f:.0f}", ha="center", fontsize=8)

# ── (1,0) Precision and Recall ───────────────────────────────────────
ax = axes[1, 0]
# noise=0.0 has no true failures so precision is undefined — skip it
valid = [i for i, n in enumerate(noise_levels) if n > 0]
nl_v = [noise_levels[i] for i in valid]
p_v = [prec_means[i] for i in valid]
p_e = [prec_stds[i] for i in valid]
r_v = [rec_means[i] for i in valid]
r_e = [rec_stds[i] for i in valid]

ax.errorbar(nl_v, p_v, yerr=p_e, color="purple", label="Precision", **kw_eb)
ax.errorbar(
    nl_v,
    r_v,
    yerr=r_e,
    color="darkorange",
    fmt="s--",
    linewidth=2,
    capsize=5,
    capthick=1.5,
    markersize=6,
    label="Recall",
)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Score  (0 = worst, 1 = best)")
ax.set_title(
    "Trigger precision and recall vs data-association noise\n"
    "Precision = fraction of triggers on real failures; "
    "Recall = fraction of failures caught"
)
ax.set_ylim(-0.05, 1.15)
ax.legend()
ax.grid(True, alpha=0.3)

# ── (1,1) Score distributions ────────────────────────────────────────
ax = axes[1, 1]
bp = ax.boxplot(
    scores_all,
    tick_labels=[str(n) for n in noise_levels],
    patch_artist=True,
    medianprops=dict(color="black", lw=2),
)
colors_bp = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(noise_levels)))
for patch, c in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(c)
ax.axhline(TAU, color="red", ls="--", lw=1.5, label=f"Trigger threshold τ = {TAU}")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Ambiguity score  log₁₀(λ_max / λ_min)  of joint covariance")
ax.set_title(
    "Per-window ambiguity score distribution\n"
    "Score insensitive to noise level → detector misses wrong-association failures"
)
ax.legend()
ax.grid(True, alpha=0.3)

# ── (1,2) Precision-Recall curve ─────────────────────────────────────
ax = axes[1, 2]
sc = ax.scatter(r_v, p_v, c=nl_v, cmap="RdYlGn_r", s=140, zorder=3, edgecolors="black")
ax.plot(r_v, p_v, "k--", alpha=0.3, lw=1)
for i, noise in zip(valid, nl_v):
    ax.annotate(f"  n={noise}", (rec_means[i], prec_means[i]), fontsize=8)
plt.colorbar(sc, ax=ax, label="Noise fraction")
ax.set_xlabel("Recall  (fraction of true failures detected)")
ax.set_ylabel("Precision  (fraction of triggers that are real failures)")
ax.set_title(
    "Precision-Recall operating curve\nIdeal detector sits at top-right (1, 1)"
)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/baseline_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/baseline_results.png")
