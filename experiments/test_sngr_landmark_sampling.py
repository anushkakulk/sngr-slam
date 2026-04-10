"""
test_sngr_landmark_sampling.py
==============================
Demonstrates refine_window_with_landmarks on the analytically bimodal
two-circle-intersection scenario, but now run
through the full SNGR pipeline rather than calling dynesty directly.

Setup
--------------------------------
Anchors at A=(0,0) and B=(4,0).  Landmark L(0) unknown.
Range measurements r_A = r_B = 3.0.
True modes: (2, +sqrt(5)) and (2, -sqrt(5)).
iSAM2 MAP: (2, 0) — the saddle between both modes, -12.5 nats.

This test shows that refine_window_with_landmarks recovers both modes
while the original refine_window (which holds landmarks fixed) cannot.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol_shorthand as sb
from scipy.stats import skew, kurtosis

from nested_sampling_refinement import (
    refine_window_with_landmarks,
    _build_mean,
    build_closure,
    extract_local_factors,
    _LandmarkLikelihood,
)

X_ = sb.X
L_ = sb.L


def build_graph():
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    range_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.2)
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
    lm_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0)  # weak — bimodal

    anchor_A = gtsam.Pose2(0.0, 0.0, 0.0)
    anchor_B = gtsam.Pose2(4.0, 0.0, 0.0)
    values.insert(X_(0), anchor_A)
    values.insert(X_(1), anchor_B)
    graph.push_back(gtsam.PriorFactorPose2(X_(0), anchor_A, pose_noise))
    graph.push_back(gtsam.PriorFactorPose2(X_(1), anchor_B, pose_noise))

    lm_init = gtsam.Point2(2.0, 0.0)
    values.insert(L_(0), lm_init)
    graph.push_back(gtsam.PriorFactorPoint2(L_(0), lm_init, lm_noise))

    r = 3.0
    graph.push_back(gtsam.RangeFactor2D(X_(0), L_(0), r, range_noise))
    graph.push_back(gtsam.RangeFactor2D(X_(1), L_(0), r, range_noise))

    return graph, values


def run():
    graph, values = build_graph()

    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    isam.update(graph, values)
    estimate = isam.calculateEstimate()
    marginals = gtsam.Marginals(isam.getFactorsUnsafe(), estimate)

    lm_map = estimate.atPoint2(L_(0))
    print(f"iSAM2 MAP for L(0): ({lm_map[0]:.4f}, {lm_map[1]:.4f})")
    print(
        f"True modes:          (2.000, +{np.sqrt(5):.3f}) and (2.000, -{np.sqrt(5):.3f})"
    )

    # wrap isam in a dummy slam_instance
    class _Slam:
        pass

    slam = _Slam()
    slam.isam = isam

    # refined_estimate = iSAM2 MAP
    refined_estimate = gtsam.Values()
    refined_estimate.insert(X_(0), estimate.atPose2(X_(0)))
    refined_estimate.insert(X_(1), estimate.atPose2(X_(1)))
    refined_estimate.insert(L_(0), estimate.atPoint2(L_(0)))

    pose_keys = [X_(0), X_(1)]
    lm_keys = [L_(0)]

    #  run refine_window_with_landmarks
    print("\nRunning refine_window_with_landmarks ...")
    mean, weights, samples = refine_window_with_landmarks(
        slam,
        refined_estimate,
        marginals,
        window_pose_keys=pose_keys,
        lm_keys_to_sample=lm_keys,
        nlive=300,
        maxiter=5000,
        cov_inflation=50.0,  # wide prior to cover both modes
        verbose=True,
    )

    # sample layout: [x0,y0,θ0, x1,y1,θ1, lx,ly]
    # landmark is at indices 6,7
    lm_samples = samples[:, 6:8]
    ess = 1.0 / np.sum(weights**2)

    closure_keys = build_closure(isam, pose_keys + lm_keys)
    factors = extract_local_factors(isam, closure_keys)
    lhood = _LandmarkLikelihood(
        factors, pose_keys, lm_keys, closure_keys, refined_estimate
    )
    logliks = np.array([lhood(s) for s in samples])
    best_idx = np.argmax(logliks)
    best_lm = lm_samples[best_idx]
    map_ll = lhood(_build_mean(refined_estimate, pose_keys, lm_keys))

    # weighted resample for bimodality coefficient
    rng = np.random.default_rng(0)
    idx_rs = rng.choice(len(samples), size=5000, p=weights)
    y_rs = lm_samples[idx_rs, 1]
    bc = (skew(y_rs) ** 2 + 1) / (kurtosis(y_rs) + 3)

    wmean = np.average(lm_samples, axis=0, weights=weights)
    wstd = np.sqrt(np.average((lm_samples - wmean) ** 2, axis=0, weights=weights))

    print(f"\n{'=' * 55}")
    print("Results")
    print(f"{'=' * 55}")
    print(
        f"iSAM2 MAP L(0):      ({lm_map[0]:.3f}, {lm_map[1]:.3f})  loglik={map_ll:.4f}"
    )
    print(
        f"Best sample L(0):    ({best_lm[0]:.3f}, {best_lm[1]:.3f})  loglik={logliks[best_idx]:.4f}"
    )
    print(f"Improvement:         {logliks[best_idx] - map_ll:+.2f} nats")
    print(f"Weighted mean L(0):  ({wmean[0]:.3f}, {wmean[1]:.3f})")
    print(
        f"Weighted std L(0):   ({wstd[0]:.3f}, {wstd[1]:.3f})  (y≈{np.sqrt(5):.3f} expected)"
    )
    print(
        f"ESS:                 {ess:.0f} / {len(samples)} ({100 * ess / len(samples):.1f}%)"
    )
    print(
        f"Bimodality coeff:    {bc:.3f}  ({'bimodal ✓' if bc > 0.555 else 'unimodal'})"
    )

    os.makedirs("results", exist_ok=True)

    # scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        lm_samples[:, 0], lm_samples[:, 1], c=weights, cmap="plasma", s=12, alpha=0.7
    )
    plt.colorbar(sc, ax=ax, label="Importance weight")
    ax.scatter(
        [2.0, 2.0],
        [np.sqrt(5), -np.sqrt(5)],
        marker="*",
        s=300,
        c="red",
        zorder=5,
        label="True modes",
    )
    ax.scatter(
        [lm_map[0]],
        [lm_map[1]],
        marker="x",
        s=200,
        c="blue",
        linewidths=2,
        zorder=5,
        label="iSAM2 MAP",
    )
    ax.scatter([0, 4], [0, 0], marker="^", s=150, c="green", zorder=5, label="Anchors")
    theta = np.linspace(0, 2 * np.pi, 300)
    for cx in [0, 4]:
        ax.plot(cx + 3 * np.cos(theta), 3 * np.sin(theta), "k--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Landmark x (m)")
    ax.set_ylabel("Landmark y (m)")
    ax.set_title("SNGR with landmark sampling\nL(0) posterior — bimodal recovery")
    ax.legend(fontsize=8)
    ax.set_xlim(-3, 7)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    plt.tight_layout()
    plt.savefig("results/sngr_landmark_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/sngr_landmark_scatter.png")

    # histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        lm_samples[:, 1],
        bins=60,
        weights=weights,
        color="steelblue",
        edgecolor="k",
        lw=0.3,
    )
    ax.axvline(np.sqrt(5), color="red", ls="--", label=f"True mode y=+{np.sqrt(5):.3f}")
    ax.axvline(
        -np.sqrt(5), color="red", ls="--", label=f"True mode y=-{np.sqrt(5):.3f}"
    )
    ax.axvline(
        lm_map[1], color="blue", ls="-", lw=2, label=f"iSAM2 MAP y={lm_map[1]:.3f}"
    )
    ax.set_xlabel("Landmark y (m)")
    ax.set_ylabel("Weighted count")
    ax.set_title(
        f"SNGR landmark posterior\nBC={bc:.3f}  ESS={100 * ess / len(samples):.0f}%  "
        f"Δloglik={logliks[best_idx] - map_ll:+.2f} nats"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/sngr_landmark_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/sngr_landmark_hist.png")


if __name__ == "__main__":
    run()
