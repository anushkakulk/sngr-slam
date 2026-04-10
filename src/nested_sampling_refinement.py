"""
refinement2.py
==============
Selective non-Gaussian refinement via Nested Sampling (dynesty).

Two entry points:
  refine_window               — samples over Pose2 window variables only
  refine_window_with_landmarks — samples over Pose2 + Point2 jointly
"""

import warnings
from typing import List, Tuple

import numpy as np
import gtsam
from dynesty import NestedSampler
from scipy.stats import norm


def poses_to_vector(estimate: gtsam.Values, keys: List[int]) -> np.ndarray:
    """Flatten Pose2 values for a list of keys into a 1-D array [x,y,θ,...]."""
    vec = []
    for k in keys:
        p = estimate.atPose2(k)
        vec.extend([p.x(), p.y(), p.theta()])
    return np.array(vec)


def vector_to_poses(x: np.ndarray, keys: List[int]) -> gtsam.Values:
    """Reconstruct a gtsam.Values from a flat sample vector (window keys only)."""
    values = gtsam.Values()
    for i, k in enumerate(keys):
        idx = 3 * i
        values.insert(k, gtsam.Pose2(x[idx], x[idx + 1], x[idx + 2]))
    return values


def _build_mean(
    estimate: gtsam.Values,
    pose_keys: List[int],
    lm_keys: List[int],
) -> np.ndarray:
    """Flat vector: [x,y,θ per pose] + [x,y per landmark]."""
    vec = []
    for k in pose_keys:
        p = estimate.atPose2(k)
        vec.extend([p.x(), p.y(), p.theta()])
    for k in lm_keys:
        pt = estimate.atPoint2(k)
        vec.extend([pt[0], pt[1]])
    return np.array(vec)


def _build_joint_cov(
    marginals: gtsam.Marginals,
    pose_keys: List[int],
    lm_keys: List[int],
) -> np.ndarray:
    """
    Block-diagonal joint covariance over poses (3x3 each) and landmarks (2x2 each).
    GTSAM's jointMarginalCovariance doesn't handle mixed Pose2/Point2 types,
    so we stack individual marginals.
    """
    blocks = []
    for k in pose_keys:
        blocks.append(marginals.marginalCovariance(k))
    for k in lm_keys:
        blocks.append(marginals.marginalCovariance(k))

    total = sum(b.shape[0] for b in blocks)
    cov = np.zeros((total, total))
    idx = 0
    for b in blocks:
        n = b.shape[0]
        cov[idx : idx + n, idx : idx + n] = b
        idx += n
    return cov


# variable closure
def build_closure(isam: gtsam.ISAM2, window_keys: List[int]) -> List[int]:
    """
    Return the full set of variable keys needed to evaluate every factor
    that touches at least one window key.
    """
    graph = isam.getFactorsUnsafe()
    keys = set(window_keys)

    changed = True
    while changed:
        changed = False
        for i in range(graph.size()):
            try:
                fk = list(graph.at(i).keys())
            except Exception:
                continue
            if any(k in keys for k in fk):
                for k in fk:
                    if k not in keys:
                        keys.add(k)
                        changed = True

    return list(keys)


# factor extraction
def extract_local_factors(isam: gtsam.ISAM2, closure_keys: List[int]) -> list:
    """Collect every factor touching at least one key in closure_keys."""
    graph = isam.getFactorsUnsafe()
    closure_set = set(closure_keys)
    factors = []
    for i in range(graph.size()):
        try:
            f = graph.at(i)
            fk = list(f.keys())
        except Exception:
            continue
        if any(k in closure_set for k in fk):
            factors.append(f)
    return factors


# pose likelihood for a local factor graph
class LocalFactorGraphLikelihood:
    """
    Log-likelihood for the local factor graph.
    Samples over window_keys (Pose2 only).
    All other closure variables are held fixed at fixed_estimate.
    """

    def __init__(
        self,
        factors: list,
        window_keys: List[int],
        closure_keys: List[int],
        fixed_estimate: gtsam.Values,
        verbose: bool = False,
    ):
        self.factors = factors
        self.window_keys = window_keys
        self.closure_keys = closure_keys
        self.fixed_estimate = fixed_estimate
        self.verbose = verbose

        window_set = set(window_keys)
        self.fixed_keys = [k for k in closure_keys if k not in window_set]
        self._warned: set = set()

    def _build_values(self, x: np.ndarray) -> gtsam.Values:
        values = vector_to_poses(x, self.window_keys)

        for k in self.fixed_keys:
            if values.exists(k):
                continue
            if not self.fixed_estimate.exists(k):
                if self.verbose and k not in self._warned:
                    warnings.warn(
                        f"[refinement] closure key {k} absent from fixed_estimate."
                    )
                    self._warned.add(k)
                continue
            try:
                values.insert(k, self.fixed_estimate.atPose2(k))
            except Exception:
                try:
                    values.insert(k, self.fixed_estimate.atPoint2(k))
                except Exception:
                    try:
                        vec = self.fixed_estimate.atVector(k)
                        values.insert(k, gtsam.Point2(vec[0], vec[1]))
                        print(f"  [debug] used atVector for {k}", flush=True)
                    except Exception:
                        if self.verbose and k not in self._warned:
                            warnings.warn(
                                f"[refinement] closure key {k}: unknown variable type, skipping."
                            )
                            self._warned.add(k)

        return values

    def log_likelihood(self, x: np.ndarray) -> float:
        try:
            values = self._build_values(x)
        except Exception as exc:
            if self.verbose:
                warnings.warn(f"[refinement] _build_values raised: {exc}")
            return -1e10

        total_error = 0.0

        for f in self.factors:
            try:
                fk = list(f.keys())
            except Exception:
                continue

            for k in fk:
                if not values.exists(k):
                    if self.verbose and (id(f), k) not in self._warned:
                        warnings.warn(
                            f"[refinement] factor {type(f).__name__} missing key {k}."
                        )
                        self._warned.add((id(f), k))
                    return -1e10

            try:
                e = float(f.error(values))
            except Exception:
                return -1e10

            if not np.isfinite(e) or abs(e) > 1e6:
                return -1e10

            total_error += e

        return -0.5 * total_error

    def __call__(self, x: np.ndarray) -> float:
        return self.log_likelihood(x)


# likelihood for poses and the landmarks
class _LandmarkLikelihood:
    """
    Log-likelihood that samples over both Pose2 and Point2 variables.

    Vector layout:
      x = [x0,y0,θ0, ..., lx0,ly0, ...]
          |<-- 3*P -->|<-- 2*L -->|
    where P = len(pose_keys), L = len(lm_keys).
    """

    def __init__(
        self,
        factors: list,
        pose_keys: List[int],
        lm_keys: List[int],
        closure_keys: List[int],
        fixed_estimate: gtsam.Values,
        verbose: bool = False,
    ):
        self.factors = factors
        self.pose_keys = pose_keys
        self.lm_keys = lm_keys
        self.closure_keys = closure_keys
        self.fixed_estimate = fixed_estimate
        self.verbose = verbose
        self.n_poses = len(pose_keys)
        self.n_lms = len(lm_keys)

        sampled = set(pose_keys) | set(lm_keys)
        self.fixed_keys = [k for k in closure_keys if k not in sampled]
        self._warned: set = set()

    def _build_values(self, x: np.ndarray) -> gtsam.Values:
        values = gtsam.Values()

        for i, k in enumerate(self.pose_keys):
            idx = 3 * i
            values.insert(k, gtsam.Pose2(x[idx], x[idx + 1], x[idx + 2]))

        offset = 3 * self.n_poses
        for i, k in enumerate(self.lm_keys):
            idx = offset + 2 * i
            values.insert(k, gtsam.Point2(x[idx], x[idx + 1]))

        for k in self.fixed_keys:
            if values.exists(k) or not self.fixed_estimate.exists(k):
                continue
            try:
                values.insert(k, self.fixed_estimate.atPose2(k))
            except Exception:
                try:
                    values.insert(k, self.fixed_estimate.atPoint2(k))
                except Exception:
                    pass

        return values

    def __call__(self, x: np.ndarray) -> float:
        try:
            values = self._build_values(x)
        except Exception:
            return -1e10

        total = 0.0
        for f in self.factors:
            try:
                fk = list(f.keys())
            except Exception:
                continue
            for k in fk:
                if not values.exists(k):
                    return -1e10
            try:
                e = float(f.error(values))
            except Exception:
                return -1e10
            if not np.isfinite(e) or abs(e) > 1e6:
                return -1e10
            total += e

        return -0.5 * total


# prior transform
def make_prior_transform(mean: np.ndarray, cov: np.ndarray):
    """Map the unit hypercube to N(mean, cov) via Cholesky."""
    cov_reg = cov + 1e-6 * np.eye(len(cov))
    L = np.linalg.cholesky(cov_reg)

    def ptform(u: np.ndarray) -> np.ndarray:
        return mean + L @ norm.ppf(u)

    return ptform


# pose only-  local NSFG refinement
def refine_window(
    slam_instance,
    isam_estimate: gtsam.Values,
    marginals: gtsam.Marginals,
    window_pose_keys: List[int],
    nlive: int = 100,
    maxiter: int = 1000,
    cov_inflation: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Nested Sampling over window_pose_keys (Pose2 only).
    Landmarks and other closure variables are held fixed at isam_estimate.
    """
    isam = slam_instance.isam

    closure_keys = build_closure(isam, window_pose_keys)
    factors = extract_local_factors(isam, closure_keys)

    if not factors:
        if verbose:
            print("[refinement] No factors found — returning MAP.")
        mean = poses_to_vector(isam_estimate, window_pose_keys)
        return mean, np.array([1.0]), mean[None]

    mean = poses_to_vector(isam_estimate, window_pose_keys)
    cov = (
        marginals.jointMarginalCovariance(
            gtsam.KeyVector(window_pose_keys)
        ).fullMatrix()
        * cov_inflation
    )

    likelihood = LocalFactorGraphLikelihood(
        factors,
        window_pose_keys,
        closure_keys,
        isam_estimate,
        verbose=verbose,
    )

    map_logp = likelihood(mean)
    if verbose or map_logp <= -1e9:
        print(
            f"[refinement] window {[gtsam.Symbol(k).string() for k in window_pose_keys]} "
            f"| MAP logprob = {map_logp:.4f}"
        )
    if map_logp <= -1e9:
        warnings.warn(
            "[refinement] MAP estimate scores -1e10: closure incomplete or NaN/Inf values. "
            "Returning MAP without refinement."
        )
        return mean, np.array([1.0]), mean[None]

    ndim = len(mean)
    sampler = NestedSampler(
        likelihood,
        make_prior_transform(mean, cov),
        ndim=ndim,
        nlive=nlive,
        sample="rwalk",
        bootstrap=0,
        walks=max(ndim, 9),
    )
    sampler.run_nested(dlogz=0.5, maxiter=maxiter, print_progress=verbose)

    res = sampler.results
    samples = res.samples
    log_weights = res.logwt - res.logz[-1]
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    posterior_mean = np.average(samples, axis=0, weights=weights)
    ess = 1.0 / np.sum(weights**2)

    best_idx = np.argmax(weights)
    best_logp = likelihood(samples[best_idx])
    map_improvement = best_logp - map_logp

    if verbose:
        print(
            f"[refinement] done | {len(samples)} samples | ESS={ess:.1f} | "
            f"logZ = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f} | "
            f"MAP Δlogp = {map_improvement:+.3f}"
        )

    return posterior_mean, weights, samples


# pose + landmark local NSFG refinement
def refine_window_with_landmarks(
    slam_instance,
    isam_estimate: gtsam.Values,
    marginals: gtsam.Marginals,
    window_pose_keys: List[int],
    lm_keys_to_sample: List[int],
    nlive: int = 100,
    maxiter: int = 1000,
    cov_inflation: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Nested Sampling over window_pose_keys (Pose2) AND lm_keys_to_sample (Point2).

    Sample vector layout:
      [x0,y0,θ0, x1,y1,θ1, ...,  lx0,ly0, lx1,ly1, ...]
      |<--- 3 * len(window_pose_keys) --->|<-- 2 * len(lm_keys_to_sample) -->|
    """
    isam = slam_instance.isam

    all_sample_keys = window_pose_keys + lm_keys_to_sample
    closure_keys = build_closure(isam, all_sample_keys)
    factors = extract_local_factors(isam, closure_keys)

    if not factors:
        if verbose:
            print("[refinement+lm] No factors found — returning MAP.")
        mean = _build_mean(isam_estimate, window_pose_keys, lm_keys_to_sample)
        return mean, np.array([1.0]), mean[None]

    mean = _build_mean(isam_estimate, window_pose_keys, lm_keys_to_sample)
    cov = (
        _build_joint_cov(marginals, window_pose_keys, lm_keys_to_sample) * cov_inflation
    )

    likelihood = _LandmarkLikelihood(
        factors,
        window_pose_keys,
        lm_keys_to_sample,
        closure_keys,
        isam_estimate,
        verbose=verbose,
    )

    map_logp = likelihood(mean)
    if verbose or map_logp <= -1e9:
        names = [gtsam.Symbol(k).string() for k in all_sample_keys]
        print(f"[refinement+lm] variables={names} | MAP logprob={map_logp:.4f}")
    if map_logp <= -1e9:
        warnings.warn("[refinement+lm] MAP scores -1e10 — returning MAP.")
        return mean, np.array([1.0]), mean[None]

    ndim = len(mean)
    sampler = NestedSampler(
        likelihood,
        make_prior_transform(mean, cov),
        ndim=ndim,
        nlive=nlive,
        sample="rwalk",
        bootstrap=0,
        walks=max(ndim, 9),
    )
    sampler.run_nested(dlogz=0.5, maxiter=maxiter, print_progress=verbose)

    res = sampler.results
    samples = res.samples
    logwt = res.logwt - res.logz[-1]
    weights = np.exp(logwt - logwt.max())
    weights /= weights.sum()

    posterior_mean = np.average(samples, axis=0, weights=weights)
    ess = 1.0 / np.sum(weights**2)

    if verbose:
        best_logp = likelihood(samples[np.argmax(weights)])
        print(
            f"[refinement+lm] done | {len(samples)} samples | ESS={ess:.1f} | "
            f"logZ={res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f} | "
            f"MAP Δlogp={best_logp - map_logp:+.3f}"
        )

    return posterior_mean, weights, samples
