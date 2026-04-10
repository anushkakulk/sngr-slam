"""
plot_refinement.py
==================
Visualises what the nested sampling refinement actually does on triggered
windows. Run on a scenario where refinement fires (noise >= 0.2).

Saves:
  results/refinement_samples.png   — sample scatter per window
  results/refinement_logp.png      — Δlogp per window
  results/refinement_ess.png       — ESS per window
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gtsam

from data_gen import build_scenario
from gaussian_slam import GaussianSLAM, X, L
from covariance_condition_number_detector import score_all_windows
from run_sngr_slam import (
    refine_window,
    build_closure,
    extract_local_factors,
    LocalFactorGraphLikelihood,
)


def run_and_refine(assoc_noise=0.2, tau=3.9, seed=0, nlive=100, maxiter=2000):
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
    triggers = list(dict.fromkeys(triggers))

    # build refined_estimate with landmarks
    refined_estimate = gtsam.Values()
    for t in range(T):
        refined_estimate.insert(X(t), estimate.atPose2(X(t)))
    for k in range(len(scenario.landmarks)):
        lk = L(k)
        if estimate.exists(lk):
            refined_estimate.insert(lk, estimate.atPoint2(lk))

    window_data = []

    for win_start in triggers:
        win_keys = pose_keys[win_start : win_start + 3]

        # MAP vector for this window
        map_vec = np.array(
            [
                c
                for k in win_keys
                for c in [
                    refined_estimate.atPose2(k).x(),
                    refined_estimate.atPose2(k).y(),
                    refined_estimate.atPose2(k).theta(),
                ]
            ]
        )

        refined_state, weights, all_samples = refine_window(
            slam,
            refined_estimate,
            marginals,
            win_keys,
            nlive=nlive,
            maxiter=maxiter,
            verbose=False,
        )

        # logp at MAP and at posterior mean
        closure_keys = build_closure(slam.isam, win_keys)
        factors = extract_local_factors(slam.isam, closure_keys)
        lhood = LocalFactorGraphLikelihood(
            factors, win_keys, closure_keys, refined_estimate
        )
        map_logp = lhood(map_vec)
        refined_logp = lhood(refined_state)
        ess = 1.0 / np.sum(weights**2)

        window_data.append(
            dict(
                win_start=win_start,
                win_keys=win_keys,
                map_vec=map_vec,
                refined_state=refined_state,
                samples=all_samples,
                weights=weights,
                map_logp=map_logp,
                refined_logp=refined_logp,
                delta_logp=refined_logp - map_logp,
                ess=ess,
                ess_frac=ess / len(all_samples),
            )
        )
        print(
            f"  Window {win_start}: Δlogp={refined_logp - map_logp:+.3f} "
            f"ESS={ess:.0f}/{len(all_samples)} ({100 * ess / len(all_samples):.1f}%)"
        )

    return scenario, window_data, triggers, scores


# sample scatter for each triggered window (x,y of each pose)
def plot_samples(scenario, window_data, assoc_noise, outpath):
    n = len(window_data)
    if n == 0:
        print("No triggered windows — nothing to plot.")
        return

    # show at most 6 windows
    window_data = window_data[:6]
    n = len(window_data)

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, wd in zip(axes, window_data):
        samples = wd["samples"]  # (N, 9) — 3 poses × (x,y,θ)
        weights = wd["weights"]
        map_vec = wd["map_vec"]
        ref_vec = wd["refined_state"]
        win_keys = wd["win_keys"]

        # plot x,y for each of the 3 poses in the window
        colors_pose = ["steelblue", "darkorange", "green"]
        for i in range(len(win_keys)):
            xi = samples[:, 3 * i]
            yi = samples[:, 3 * i + 1]
            ax.scatter(xi, yi, c=weights, cmap="plasma", s=6, alpha=0.5, zorder=2)
            # MAP point
            ax.scatter(
                map_vec[3 * i],
                map_vec[3 * i + 1],
                marker="x",
                s=120,
                c=colors_pose[i],
                linewidths=2,
                zorder=5,
                label=f"MAP x{gtsam.Symbol(win_keys[i]).index()}",
            )
            # posterior mean
            ax.scatter(
                ref_vec[3 * i],
                ref_vec[3 * i + 1],
                marker="o",
                s=80,
                c=colors_pose[i],
                edgecolors="black",
                linewidths=1,
                zorder=5,
                label=f"Mean x{gtsam.Symbol(win_keys[i]).index()}",
            )

        # ground truth poses for this window
        for i, k in enumerate(win_keys):
            t = gtsam.Symbol(k).index()
            gt = scenario.poses[t, :2]
            ax.scatter(*gt, marker="*", s=150, c="red", zorder=6)

        t0 = gtsam.Symbol(win_keys[0]).index()
        ax.set_title(
            f"Window {wd['win_start']} (x{t0}–x{t0 + 2})\n"
            f"Δlogp={wd['delta_logp']:+.2f}  "
            f"ESS={wd['ess_frac'] * 100:.0f}%",
            fontsize=9,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.2)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Nested sampling posterior samples per triggered window\n"
        f"noise={assoc_noise}  |  × = MAP,  ● = posterior mean,  ★ = ground truth",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


# Δlogp per window
def plot_delta_logp(window_data, tau, outpath):
    if not window_data:
        return
    wins = [wd["win_start"] for wd in window_data]
    deltas = [wd["delta_logp"] for wd in window_data]
    colors = ["green" if d >= 0 else "gray" for d in deltas]

    fig, ax = plt.subplots(figsize=(max(8, len(wins) * 0.6), 4))
    bars = ax.bar(wins, deltas, color=colors, width=0.7, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(
        0.2, color="gray", ls="--", lw=1, label="Sampling noise threshold (~0.2 nats)"
    )
    ax.set_xlabel("Window start index")
    ax.set_ylabel("Δlogp  (posterior mean − iSAM2 MAP)")
    ax.set_title(
        "Log-probability improvement from nested sampling refinement\n"
        "Green = sampler found better pose than MAP  |  "
        "Gray = MAP retained"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    for bar, d in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + (0.05 if d >= 0 else -0.15),
            f"{d:+.2f}",
            ha="center",
            fontsize=7,
        )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


# Plot 3: ESS per window
def plot_ess(window_data, outpath):
    if not window_data:
        return
    wins = [wd["win_start"] for wd in window_data]
    ess_fracs = [wd["ess_frac"] * 100 for wd in window_data]
    colors = ["coral" if e < 20 else "steelblue" for e in ess_fracs]

    fig, ax = plt.subplots(figsize=(max(8, len(wins) * 0.6), 4))
    ax.bar(wins, ess_fracs, color=colors, width=0.7, alpha=0.85)
    ax.axhline(20, color="red", ls="--", lw=1.5, label="ESS=20% — bimodality threshold")
    ax.axhline(
        50, color="green", ls="--", lw=1, label="ESS=50% — approximately Gaussian"
    )
    ax.set_xlabel("Window start index")
    ax.set_ylabel("Effective Sample Size  (%  of total samples)")
    ax.set_title(
        "Posterior ESS per triggered window\n"
        "Low ESS → non-Gaussian / multimodal posterior  |  "
        "High ESS → Gaussian posterior, MAP is reliable"
    )
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


if __name__ == "__main__":
    NOISE = 0.2
    TAU = 3.9
    SEED = 0
    NLIVE = 100
    MAXITER = 2000

    print(f"Running pipeline: noise={NOISE}, tau={TAU}, seed={SEED}")
    scenario, window_data, triggers, scores = run_and_refine(
        assoc_noise=NOISE,
        tau=TAU,
        seed=SEED,
        nlive=NLIVE,
        maxiter=MAXITER,
    )

    os.makedirs("results", exist_ok=True)
    plot_samples(scenario, window_data, NOISE, "results/refinement_samples.png")
    plot_delta_logp(window_data, TAU, "results/refinement_logp.png")
    plot_ess(window_data, "results/refinement_ess.png")
