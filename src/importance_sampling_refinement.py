# ============================================================
"""
Non-Gaussian refinement for a flagged local window.
Strategy: enumerate a small set of data-association hypotheses,
weight each by its Gaussian marginal likelihood, return the MAP
hypothesis and its weight.

For range-only SLAM without explicit data assoc ambiguity,
this falls back to a simple rejection-sampling posterior estimate
over the window poses.
"""

import numpy as np
from scipy.stats import multivariate_normal
import gtsam


def gaussian_log_likelihood(mean, cov, x):
    return multivariate_normal.logpdf(x, mean=mean, cov=cov)


def hypothesis_weight(marginals, gt_mean, gt_cov, hypothesis_poses):
    """
    Compute unnormalised log weight for a single hypothesis
    as the Gaussian likelihood of the hypothesis poses under
    the current marginal.
    """
    return gaussian_log_likelihood(gt_mean, gt_cov, hypothesis_poses)


def rejection_sample_window(
    isam_estimate, marginals, window_pose_keys, n_samples=500, sigma_proposal=0.5
):
    """
    Simple rejection sampler around the MAP estimate for a local window.
    Used as a lightweight stand-in for nested sampling.

    Returns:
      samples  : (n_samples, dim) array of accepted samples
      log_weights: (n_samples,) importance weights
    """
    keys = gtsam.KeyVector()
    for k in window_pose_keys:
        keys.append(k)
    cov = marginals.jointMarginalCovariance(keys).fullMatrix()
    mean = np.array(
        [
            [
                isam_estimate.atPose2(k).x(),
                isam_estimate.atPose2(k).y(),
                isam_estimate.atPose2(k).theta(),
            ]
            for k in window_pose_keys
        ]
    ).flatten()
    proposal = multivariate_normal(mean=mean, cov=sigma_proposal**2 * np.eye(len(mean)))
    target = multivariate_normal(mean=mean, cov=cov)

    samples = proposal.rvs(n_samples)
    log_w = target.logpdf(samples) - proposal.logpdf(samples)
    return samples, log_w


def refine_window(isam_estimate, marginals, window_pose_keys, n_samples=500):
    """
    Run local non-Gaussian refinement on a flagged window.
    Returns the best-weight pose sample as a refined estimate.
    """
    samples, log_w = rejection_sample_window(
        isam_estimate, marginals, window_pose_keys, n_samples
    )
    best_idx = np.argmax(log_w)
    return samples[best_idx], log_w
