import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run_sngr_slam import run

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
seeds = [0, 1, 2, 3, 4]
TAU = 3.9

rmse_means, rmse_stds = [], []
lm_rmse_means, lm_rmse_stds = [], []
nees_means, nees_stds = [], []
prec_means, prec_stds = [], []
rec_means, rec_stds = [], []
trigger_means = []
t_gauss_means = []
t_ref_means, t_ref_stds = [], []
scores_all = []

for noise in noise_levels:
    rmses, lm_rmses, neess = [], [], []
    precs, recs, trigs = [], [], []
    t_gausses, t_refs = [], []
    scs = []

    for seed in seeds:
        rmse, mean_nees, lm_rmse, t_gauss, t_ref, pr = run(
            assoc_noise=noise,
            tau=TAU,
            nlive=75,
            maxiter=500,
            seed=seed,
            verbose=True,
        )
        rmses.append(rmse)
        lm_rmses.append(lm_rmse)
        neess.append(mean_nees)
        precs.append(pr["precision"] if pr["precision"] is not None else float("nan"))
        recs.append(pr["recall"] if pr["recall"] is not None else float("nan"))
        trigs.append(pr["tp"] + pr["fp"])  # total triggers
        t_gausses.append(t_gauss)
        t_refs.append(t_ref)

    rmse_means.append(np.mean(rmses))
    rmse_stds.append(np.std(rmses))
    lm_rmse_means.append(np.mean(lm_rmses))
    lm_rmse_stds.append(np.std(lm_rmses))
    nees_means.append(np.mean(neess))
    nees_stds.append(np.std(neess))
    prec_means.append(np.nanmean(precs))
    prec_stds.append(np.nanstd(precs))
    rec_means.append(np.nanmean(recs))
    rec_stds.append(np.nanstd(recs))
    trigger_means.append(np.mean(trigs))
    t_gauss_means.append(np.mean(t_gausses))
    t_ref_means.append(np.mean(t_refs))
    t_ref_stds.append(np.std(t_refs))

    print(
        f"noise={noise:.1f} | "
        f"pose RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f} | "
        f"NEES={np.mean(neess):.1f} | "
        f"P={np.nanmean(precs):.2f} R={np.nanmean(recs):.2f} | "
        f"triggers={np.mean(trigs):.1f}/28 | "
        f"t_ref={np.mean(t_refs):.1f}s"
    )


fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle(
    "Selective Non-Gaussian Refinement — Full Results\n"
    f"iSAM2 + condition-number trigger (τ={TAU}) + nested sampling refinement",
    fontsize=13,
)

kw_eb = dict(fmt="o-", linewidth=2, capsize=5, capthick=1.5, markersize=6)

# post RMSE
ax = axes[0, 0]
ax.errorbar(
    noise_levels,
    rmse_means,
    yerr=rmse_stds,
    color="steelblue",
    label="Pose RMSE",
    **kw_eb,
)
ax.axhline(0.5, color="gray", ls="--", alpha=0.6, label="0.5 m threshold")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Pose RMSE (m)")
ax.set_title("Robot pose accuracy vs data-association noise")
ax.legend()
ax.grid(True, alpha=0.3)

# landmark RMSE
ax = axes[0, 1]
ax.errorbar(
    noise_levels,
    lm_rmse_means,
    yerr=lm_rmse_stds,
    color="teal",
    label="Landmark RMSE",
    **kw_eb,
)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Landmark RMSE (m)")
ax.set_title("Landmark accuracy vs data-association noise")
ax.legend()
ax.grid(True, alpha=0.3)

# NEES
ax = axes[0, 2]
ax.errorbar(
    noise_levels, nees_means, yerr=nees_stds, color="coral", label="Mean NEES", **kw_eb
)
ax.axhline(2.0, color="green", ls="--", lw=1.5, label="Consistent (NEES = 2)")
ax.set_yscale("log")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Mean NEES (log scale)")
ax.set_title(
    "Filter consistency vs data-association noise\n"
    "NEES ≫ 2 → overconfident; NEES ≪ 2 → underconfident"
)
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# Trigger count
ax = axes[1, 0]
colors = ["steelblue" if t < 1 else "coral" for t in trigger_means]
bars = ax.bar(noise_levels, trigger_means, width=0.07, color=colors)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Mean windows triggered (out of 28)")
ax.set_title(
    f"Trigger rate vs data-association noise (τ = {TAU})\n"
    "Blue = no triggers; coral = triggers fired"
)
ax.set_ylim(0, 30)
ax.grid(True, alpha=0.3, axis="y")
for x, y in zip(noise_levels, trigger_means):
    ax.text(x, y + 0.4, f"{y:.1f}", ha="center", fontsize=9)

# score distribution
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
ax.axhline(TAU, color="red", ls="--", lw=1.5, label=f"Trigger threshold τ={TAU}")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Window ambiguity score  log₁₀(κ(Σ_w))")
ax.set_title(
    "Distribution of condition-number scores per noise level\n"
    "Windows above red line trigger non-Gaussian refinement"
)
ax.legend()
ax.grid(True, alpha=0.3)

# precision and recall
ax = axes[1, 2]
ax.errorbar(
    noise_levels[1:],
    prec_means[1:],
    yerr=prec_stds[1:],
    color="purple",
    label="Precision",
    **kw_eb,
)
ax.errorbar(
    noise_levels[1:],
    rec_means[1:],
    yerr=rec_stds[1:],
    fmt="s--",
    color="darkorange",
    linewidth=2,
    capsize=5,
    capthick=1.5,
    markersize=6,
    label="Recall",
)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Score")
ax.set_title(
    "Trigger precision and recall vs data-association noise\n"
    "True positive = trigger on a genuinely high-error window"
)
ax.set_ylim(-0.05, 1.15)
ax.legend()
ax.grid(True, alpha=0.3)

# precision-recall scatter
ax = axes[2, 0]
sc = ax.scatter(
    rec_means,
    prec_means,
    c=noise_levels,
    cmap="RdYlGn_r",
    s=140,
    zorder=3,
    edgecolors="black",
)
ax.plot(rec_means, prec_means, "k--", alpha=0.3, lw=1)
for i, noise in enumerate(noise_levels):
    ax.annotate(f" n={noise}", (rec_means[i], prec_means[i]), fontsize=8)
plt.colorbar(sc, ax=ax, label="Noise fraction")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall operating point per noise level")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

# ── (2,1) Wall-clock time ────────────────────────────────────────────
ax = axes[2, 1]
width = 0.035
x = np.array(noise_levels)
bars_g = ax.bar(
    x - width, t_gauss_means, width=width * 2, color="steelblue", label="iSAM2"
)
bars_r = ax.bar(
    x + width,
    t_ref_means,
    width=width * 2,
    color="coral",
    label="Refinement (triggered windows)",
)
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title(
    "Computation time: iSAM2 vs selective refinement\n"
    "Refinement only runs on triggered windows"
)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# cost per triggered window (refinement time / trigger count)
ax = axes[2, 2]
# cost per window = total refinement time / number of triggers (avoid div/0)
cost_per_window = [t / max(n, 1) for t, n in zip(t_ref_means, trigger_means)]
ax.bar(noise_levels, cost_per_window, width=0.07, color="mediumpurple")
ax.set_xlabel("Wrong-association noise fraction")
ax.set_ylabel("Mean time per triggered window (s)")
ax.set_title(
    "Per-window refinement cost\n"
    "Constant cost shows sampler scales with window size, not noise"
)
ax.grid(True, alpha=0.3, axis="y")
for x_, y in zip(noise_levels, cost_per_window):
    if y > 0:
        ax.text(x_, y + 0.2, f"{y:.1f}s", ha="center", fontsize=9)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/results_selective_nsfg.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/results_selective_nsfg.png")
