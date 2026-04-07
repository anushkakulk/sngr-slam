"""
test_bimodal.py
===============
Proof-of-concept: nested sampling finds a bimodal posterior in a
range-only factor graph where iSAM2 fails.

Setup
-----
Two anchor poses at known positions: A=(0,0), B=(4,0).
One landmark L(0) at unknown position.
Two range measurements: r_A = r_B = 3.0.

The intersection of two circles of radius 3 centred at (0,0) and (4,0)
gives exactly two points: (2, +sqrt(5)) and (2, -sqrt(5)) ~= (2, +-2.236).
The posterior over L(0) is bimodal -- analytically guaranteed.

iSAM2 initialises at the midpoint (2, 0) -- the saddle between modes --
and converges there with log-likelihood ~-12.5. Nested sampling finds
both modes and recovers a best sample within 0.01 of the true mode,
a +12.49 nat improvement.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol_shorthand as sb
from dynesty import NestedSampler
from scipy.stats import norm, skew, kurtosis

X_ = sb.X
L_ = sb.L


def build_graph():
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    range_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.2)
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
    lm_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0)

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


def make_loglik(graph, fixed_values):
    def loglik(xy):
        values = gtsam.Values(fixed_values)
        if values.exists(L_(0)):
            values.update(L_(0), gtsam.Point2(xy[0], xy[1]))
        else:
            values.insert(L_(0), gtsam.Point2(xy[0], xy[1]))
        try:
            e = graph.error(values)
            return -0.5 * e if np.isfinite(e) else -1e10
        except Exception:
            return -1e10

    return loglik


def make_prior_transform(mean, std):
    def ptform(u):
        return mean + std * norm.ppf(u)

    return ptform


def plot_samples(samples, weights, lm_map):
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        samples[:, 0], samples[:, 1], c=weights, cmap="plasma", s=15, alpha=0.7
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
        ax.plot(cx + 3 * np.cos(theta), 3 * np.sin(theta), "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Landmark x")
    ax.set_ylabel("Landmark y")
    ax.set_title(
        "Nested sampling — L(0) posterior\n(bimodal: two circle intersections)"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(-3, 7)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    plt.tight_layout()
    plt.savefig("bimodal_samples.png", dpi=150)
    print("Scatter plot saved to bimodal_samples.png")


def plot_histogram(samples, weights):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        samples[:, 1],
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
    ax.set_xlabel("Landmark y")
    ax.set_ylabel("Weighted count")
    ax.set_title("Bimodal posterior — two peaks at y=+-sqrt(5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("bimodal_hist.png", dpi=150)
    print("Histogram saved to bimodal_hist.png")


def run_bimodal_test():
    graph, values = build_graph()

    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    isam.update(graph, values)
    estimate = isam.calculateEstimate()
    lm_map = estimate.atPoint2(L_(0))

    fixed = gtsam.Values()
    fixed.insert(X_(0), estimate.atPose2(X_(0)))
    fixed.insert(X_(1), estimate.atPose2(X_(1)))

    loglik = make_loglik(graph, fixed)
    map_ll = loglik(np.array([lm_map[0], lm_map[1]]))
    true_ll = loglik(np.array([2.0, np.sqrt(5)]))

    print(f"iSAM2 MAP: ({lm_map[0]:.4f}, {lm_map[1]:.4f})  loglik={map_ll:.4f}")
    print(f"True modes: (2.000, +-{np.sqrt(5):.3f})  loglik={true_ll:.4f}")
    print(f"iSAM2 is {true_ll - map_ll:.2f} nats below the true mode\n")

    ptform = make_prior_transform(mean=np.array([2.0, 0.0]), std=np.array([4.0, 4.0]))
    sampler = NestedSampler(
        loglik, ptform, ndim=2, nlive=300, sample="rwalk", bootstrap=0, walks=5
    )
    sampler.run_nested(dlogz=0.1, print_progress=True)

    res = sampler.results
    samples = res.samples
    logwt = res.logwt - res.logz[-1]
    weights = np.exp(logwt - logwt.max())
    weights /= weights.sum()

    ess = 1.0 / np.sum(weights**2)
    wmean = np.average(samples, axis=0, weights=weights)
    wstd = np.sqrt(np.average((samples - wmean) ** 2, axis=0, weights=weights))

    logliks = np.array([loglik(s) for s in samples])
    best_idx = np.argmax(logliks)
    best_ll = logliks[best_idx]
    best_xy = samples[best_idx]

    rng = np.random.default_rng(0)
    idx_rs = rng.choice(len(samples), size=5000, p=weights)
    y_rs = samples[idx_rs, 1]
    bc = (skew(y_rs) ** 2 + 1) / (kurtosis(y_rs) + 3)

    print(
        f"\nSamples: {len(samples)} | ESS: {ess:.1f} ({100 * ess / len(samples):.1f}%)"
    )
    print(f"logZ: {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}")
    print(
        f"Weighted mean: ({wmean[0]:.3f}, {wmean[1]:.3f})  std: ({wstd[0]:.3f}, {wstd[1]:.3f})"
    )
    print(f"Best sample: ({best_xy[0]:.3f}, {best_xy[1]:.3f})  loglik={best_ll:.4f}")
    print(f"Improvement over iSAM2 MAP: {best_ll - map_ll:+.2f} nats")
    print(
        f"Bimodality coefficient (weighted): {bc:.3f} ({'bimodal' if bc > 0.555 else 'unimodal'})"
    )

    plot_samples(samples, weights, lm_map)
    plot_histogram(samples, weights)


if __name__ == "__main__":
    run_bimodal_test()
