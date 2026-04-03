import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run import run

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
seeds = [0, 1, 2, 3, 4]
TAU = 3.96

# results collection across seeds
rmse_means, rmse_stds = [], []
nees_means, nees_stds = [], []
prec_means, prec_stds = [], []
rec_means, rec_stds = [], []
trigger_means = []
scores_all = []

for noise in noise_levels:
    rmses_s, nees_s, prec_s, rec_s, trig_s, sc_s = [], [], [], [], [], []
    for seed in seeds:
        r = run(assoc_noise=noise, tau=TAU, seed=seed, verbose=False)
        # print(f"  seed={seed} raw prec={r['precision']} type={type(r['precision'])}")
        rmses_s.append(r["rmse"])
        nees_s.append(r["nees"])
        prec_s.append(
            float(r["precision"])
            if not np.isnan(float(r["precision"]))
            else float("nan")
        )
        rec_s.append(
            float(r["recall"]) if not np.isnan(float(r["recall"])) else float("nan")
        )
        trig_s.append(len(r["triggers"]))
        sc_s.extend([s for _, s in r["scores"]])
    rmse_means.append(np.mean(rmses_s))
    rmse_stds.append(np.std(rmses_s))
    nees_means.append(np.mean(nees_s))
    nees_stds.append(np.std(nees_s))
    if not all(np.isnan(prec_s)):
        prec_means.append(np.nanmean(prec_s))
        prec_stds.append(np.nanstd(prec_s))
    else:
        prec_means.append(float("nan"))
        prec_stds.append(0.0)
    rec_means.append(np.nanmean(rec_s))
    rec_stds.append(np.nanstd(rec_s))
    trigger_means.append(np.mean(trig_s))
    scores_all.append(sc_s)

    print(
        f"noise={noise} | RMSE={np.mean(rmses_s):.3f}±{np.std(rmses_s):.3f} "
        f"| NEES={np.mean(nees_s):.1f}±{np.std(nees_s):.1f} "
        f"| P={np.nanmean(prec_s):.3f} R={np.nanmean(rec_s):.3f} "
        f"| triggers={np.mean(trig_s):.1f}/28"
    )

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Selective Non-Gaussian Refinement — Results", fontsize=13)

# RMSE vs noise
ax = axes[0, 0]
ax.errorbar(
    noise_levels,
    rmse_means,
    yerr=rmse_stds,
    fmt="o-",
    color="steelblue",
    linewidth=2,
    capsize=5,
    capthick=1.5,
    label="mean ± std",
)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5m threshold")
ax.set_xlabel("Association noise fraction")
ax.set_ylabel("Pose RMSE (m)")
ax.set_title("iSAM2 RMSE vs ambiguity")
ax.legend()
ax.grid(True, alpha=0.3)

# NEES vs noise (log scale — values span 0.3 to 745)
ax = axes[0, 1]
ax.errorbar(
    noise_levels,
    nees_means,
    yerr=nees_stds,
    fmt="s-",
    color="coral",
    linewidth=2,
    capsize=5,
    capthick=1.5,
    label="mean ± std",
)
ax.axhline(
    y=2.0, color="green", linestyle="--", linewidth=1.5, label="consistent (NEES=2)"
)
ax.set_yscale("log")
ax.set_xlabel("Association noise fraction")
ax.set_ylabel("Mean NEES (log scale)")
ax.set_title("Estimator consistency vs ambiguity")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# Trigger count vs noise
ax = axes[0, 2]
bar_colors = ["green" if t < 0.5 else "coral" for t in trigger_means]
ax.bar(noise_levels, trigger_means, width=0.07, color=bar_colors)
ax.set_xlabel("Association noise fraction")
ax.set_ylabel("Mean windows triggered")
ax.set_title(f"Trigger count vs ambiguity (τ={TAU})")
ax.set_ylim(0, 30)
ax.grid(True, alpha=0.3, axis="y")
# label each bar
for x, y in zip(noise_levels, trigger_means):
    ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=9)

# Precision vs noise (skip noise=0.0: undefined)
ax = axes[1, 0]
ax.errorbar(
    noise_levels[1:],
    prec_means[1:],
    yerr=prec_stds[1:],
    fmt="o-",
    color="purple",
    linewidth=2,
    capsize=5,
    capthick=1.5,
    label="precision",
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
    label="recall",
)
ax.set_xlabel("Association noise fraction")
ax.set_ylabel("Score")
ax.set_title("Trigger precision and recall vs ambiguity")
ax.set_ylim(-0.05, 1.15)
ax.legend()
ax.grid(True, alpha=0.3)

# Score distributions
ax = axes[1, 1]
bp = ax.boxplot(
    scores_all, tick_labels=[str(n) for n in noise_levels], patch_artist=True
)
colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(noise_levels)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.axhline(y=TAU, color="red", linestyle="--", linewidth=1.5, label=f"τ={TAU}")
ax.set_xlabel("Association noise fraction")
ax.set_ylabel("Condition number score κ(H_w)")
ax.set_title("Score distribution vs ambiguity")
ax.legend()
ax.grid(True, alpha=0.3)

# Precision-Recall curve across noise levels
ax = axes[1, 2]
sc = ax.scatter(
    rec_means,
    prec_means,
    c=noise_levels,
    cmap="RdYlGn_r",
    s=120,
    zorder=3,
    edgecolors="black",
)
ax.plot(rec_means, prec_means, "k--", alpha=0.3, linewidth=1)
for i, noise in enumerate(noise_levels):
    ax.annotate(f"  n={noise}", (rec_means[i], prec_means[i]), fontsize=8)
plt.colorbar(sc, ax=ax, label="Assoc. noise")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/results_full.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results_full.png")
