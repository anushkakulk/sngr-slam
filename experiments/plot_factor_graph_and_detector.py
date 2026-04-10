"""
plot_gaussian_factor_graph.py
====================
Visualises:
  1. The SLAM factor graph (poses, landmarks, odometry edges, range edges)
     with iSAM2 estimates overlaid on ground truth.
  2. The per-window ambiguity score with triggered windows highlighted.
  3. Side-by-side comparison of clean vs corrupted scenarios.

"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from data_gen import build_scenario
from gaussian_slam import GaussianSLAM, X, L
from covariance_condition_number_detector import score_all_windows


def run_and_collect(assoc_noise, tau, seed=0):
    scenario = build_scenario(T=30, K=6, assoc_noise=assoc_noise, seed=seed)
    T = len(scenario.poses)

    slam = GaussianSLAM(sigma_range=scenario.range_noise)
    slam.initialise(scenario.poses[0])

    meas_by_t = {t: [] for t in range(T)}
    for t, k, r in scenario.range_meas:
        meas_by_t[t].append((k, r))

    for t in range(1, T):
        slam.step(t, scenario.odometry[t - 1], meas_by_t[t], scenario.landmarks.copy())

    marginals, estimate = slam.get_marginals()

    pose_keys = [X(t) for t in range(T)]
    lm_keys = [L(k) for k in range(len(scenario.landmarks))]

    scores, triggers = score_all_windows(
        marginals, pose_keys, lm_keys, tau=tau, meas_by_t=meas_by_t
    )

    # extract estimated poses and landmarks
    est_poses = np.array(
        [[estimate.atPose2(X(t)).x(), estimate.atPose2(X(t)).y()] for t in range(T)]
    )
    est_lms = []
    for k in range(len(scenario.landmarks)):
        if estimate.exists(L(k)):
            pt = estimate.atPoint2(L(k))
            est_lms.append([pt[0], pt[1]])
        else:
            est_lms.append([np.nan, np.nan])
    est_lms = np.array(est_lms)

    return dict(
        scenario=scenario,
        est_poses=est_poses,
        est_lms=est_lms,
        scores=scores,
        triggers=triggers,
        meas_by_t=meas_by_t,
        T=T,
    )


def plot_factor_graph(ax, data, title):
    scenario = data["scenario"]
    est_poses = data["est_poses"]
    est_lms = data["est_lms"]
    meas_by_t = data["meas_by_t"]
    triggers = set(data["triggers"])
    T = data["T"]
    gt_poses = scenario.poses
    gt_lms = scenario.landmarks

    for t in range(T):
        for k, _ in meas_by_t.get(t, []):
            if k < len(est_lms) and not np.isnan(est_lms[k, 0]):
                ax.plot(
                    [est_poses[t, 0], est_lms[k, 0]],
                    [est_poses[t, 1], est_lms[k, 1]],
                    color="lightcoral",
                    lw=0.4,
                    alpha=0.4,
                    zorder=1,
                )

    for t in range(T - 1):
        ax.plot(
            [est_poses[t, 0], est_poses[t + 1, 0]],
            [est_poses[t, 1], est_poses[t + 1, 1]],
            color="steelblue",
            lw=1.2,
            alpha=0.7,
            zorder=2,
        )

    ax.plot(
        gt_poses[:, 0],
        gt_poses[:, 1],
        "k--",
        lw=1.0,
        alpha=0.4,
        zorder=2,
        label="Ground truth",
    )
    ax.plot(
        np.append(gt_poses[:, 0], gt_poses[0, 0]),
        np.append(gt_poses[:, 1], gt_poses[0, 1]),
        "k--",
        lw=1.0,
        alpha=0.4,
        zorder=2,
    )

    triggered_ts = set()
    for win_start in triggers:
        for tt in range(win_start, min(win_start + 3, T)):
            triggered_ts.add(tt)

    normal_ts = [t for t in range(T) if t not in triggered_ts]
    triggered_ts = sorted(triggered_ts)

    ax.scatter(
        est_poses[normal_ts, 0],
        est_poses[normal_ts, 1],
        c="steelblue",
        s=30,
        zorder=4,
        label="Estimated pose",
    )
    if triggered_ts:
        ax.scatter(
            est_poses[triggered_ts, 0],
            est_poses[triggered_ts, 1],
            c="red",
            s=50,
            zorder=5,
            label="Triggered window pose",
            edgecolors="darkred",
            linewidths=0.8,
        )

    # landmarks
    ax.scatter(
        gt_lms[:, 0],
        gt_lms[:, 1],
        marker="*",
        s=200,
        c="gold",
        edgecolors="darkorange",
        linewidths=0.8,
        zorder=5,
        label="Landmark (GT)",
    )
    valid = ~np.isnan(est_lms[:, 0])
    ax.scatter(
        est_lms[valid, 0],
        est_lms[valid, 1],
        marker="+",
        s=100,
        c="darkorange",
        linewidths=1.5,
        zorder=5,
        label="Landmark (est)",
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)


def plot_scores(ax, data, tau, title):
    scores = data["scores"]
    triggers = set(data["triggers"])

    ts = [t for t, _ in scores]
    vals = [s for _, s in scores]
    colors = ["red" if t in triggers else "steelblue" for t in ts]

    ax.bar(ts, vals, color=colors, width=0.8, alpha=0.8)
    ax.axhline(tau, color="red", ls="--", lw=1.5, label=f"Threshold τ={tau}")
    ax.set_xlabel("Window start index")
    ax.set_ylabel("Ambiguity score  log₁₀(λ_max/λ_min)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # legend patches
    triggered_patch = mpatches.Patch(color="red", alpha=0.8, label="Triggered")
    normal_patch = mpatches.Patch(color="steelblue", alpha=0.8, label="Not triggered")
    ax.legend(
        handles=[
            triggered_patch,
            normal_patch,
            Line2D([0], [0], color="red", ls="--", lw=1.5, label=f"τ = {tau}"),
        ],
        fontsize=8,
    )


TAU = 3.96
SEED = 0

data_clean = run_and_collect(assoc_noise=0.0, tau=TAU, seed=SEED)
data_noisy = run_and_collect(assoc_noise=0.3, tau=TAU, seed=SEED)

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 8))
plot_factor_graph(
    ax,
    data_clean,
    "Factor graph — clean data (noise=0.0)\n"
    "Red poses = triggered windows  |  Gold stars = GT landmarks",
)
plt.tight_layout()
plt.savefig("results/gaussian_factor_graph_clean.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/gaussian_factor_graph_clean.png")

#  corrupted factor graph
fig, ax = plt.subplots(figsize=(8, 8))
plot_factor_graph(
    ax,
    data_noisy,
    "Factor graph — corrupted data (noise=0.3)\n"
    "Red poses = triggered windows  |  Gold stars = GT landmarks",
)
plt.tight_layout()
plt.savefig("results/gaussian_factor_graph_noisy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/gaussian_factor_graph_noisy.png")

# clean factor graph - covariance condition numebr detector scores
fig, ax = plt.subplots(figsize=(10, 4))
plot_scores(
    ax,
    data_clean,
    TAU,
    "Ambiguity scores — clean data (noise=0.0)\n"
    "Score measures covariance condition number per window",
)
plt.tight_layout()
plt.savefig("results/detector_scores_clean.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/detector_scores_clean.png")

# corrupted factor graph - covariance condition numebr detector scores
fig, ax = plt.subplots(figsize=(10, 4))
plot_scores(
    ax,
    data_noisy,
    TAU,
    "Ambiguity scores — corrupted data (noise=0.3)\n"
    "More windows exceed threshold as wrong associations distort covariance",
)
plt.tight_layout()
plt.savefig("results/detector_scores_noisy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/detector_scores_noisy.png")
