"""
data_gen.py
===========
Two distinct experimental manipulations, each targeting a different
failure mode of Gaussian SLAM:

1. assoc_noise  — wrong data association (outlier measurements).
   A fraction of range factors point to the wrong landmark.
   Effect: some factors are geometrically irreconcilable → high residual error.
   iSAM2 response: biased estimate, inflated covariance.
   Refinement role: robust cost function / outlier rejection.

2. range_ambiguity — genuine multimodality from symmetric landmark geometry.
   Landmarks are placed in pairs equidistant from the trajectory so that
   range measurements alone cannot distinguish which side the robot is on.
   Effect: bimodal position posterior.
   iSAM2 response: collapses to one mode, underestimates uncertainty.
   Refinement role: Nested Sampling discovers both modes.

These are kept as separate knobs so experiments can isolate each effect.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SLAMScenario:
    poses: np.ndarray  # (T, 3)  ground-truth [x, y, theta]
    landmarks: np.ndarray  # (K, 2)  ground-truth [x, y]
    odometry: np.ndarray  # (T-1, 3) noisy local-frame odometry
    range_meas: List[Tuple]  # [(t, k, range), ...]
    assoc_noise: float = 0.0
    range_noise: float = 0.1
    ambiguity_pairs: List = field(default_factory=list)
    # ambiguity_pairs: list of (k_a, k_b) landmark index pairs that are
    # symmetric w.r.t. the trajectory — the ground-truth partner of any
    # mis-associated measurement.  Used by the evaluator.


# ------------------------------------------------------------------ #
# Trajectory                                                          #
# ------------------------------------------------------------------ #


def generate_circular_trajectory(T=30, radius=5.0, seed=0):
    angles = np.linspace(0, 2 * np.pi, T, endpoint=False)
    poses = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), angles + np.pi / 2]
    )
    return poses


# ------------------------------------------------------------------ #
# Odometry                                                            #
# ------------------------------------------------------------------ #


def add_odometry_noise(poses, sigma_t=0.05, sigma_r=0.01, seed=1):
    rng = np.random.default_rng(seed)
    T = len(poses)
    odom = []
    for i in range(T - 1):
        dx_world = poses[i + 1, 0] - poses[i, 0]
        dy_world = poses[i + 1, 1] - poses[i, 1]
        theta = poses[i, 2]
        dx_local = np.cos(theta) * dx_world + np.sin(theta) * dy_world
        dy_local = -np.sin(theta) * dx_world + np.cos(theta) * dy_world
        dtheta = poses[i + 1, 2] - poses[i, 2]
        noise = rng.normal([0, 0, 0], [sigma_t, sigma_t, sigma_r])
        odom.append([dx_local + noise[0], dy_local + noise[1], dtheta + noise[2]])
    return np.array(odom)


# ------------------------------------------------------------------ #
# Landmark layouts                                                    #
# ------------------------------------------------------------------ #


def random_landmarks(K, spread=4.0, seed=0):
    """Fully random layout — baseline, no intentional ambiguity."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, (K, 2)), []


def symmetric_landmarks(K_pairs, radius, spread=3.0, seed=0):
    """
    Place K_pairs pairs of landmarks symmetric about the origin.
    For a circular trajectory centred at the origin, both landmarks in a
    pair are equidistant from every robot pose, so range measurements
    are consistent with the robot being at the true pose OR its mirror.

    This is the canonical setup that creates a bimodal posterior:
    the Gaussian solver collapses onto one mode while the non-Gaussian
    inference should maintain probability mass on both.
    """
    rng = np.random.default_rng(seed)
    lms = []
    pairs = []
    for i in range(K_pairs):
        # Random angle for the pair axis
        angle = rng.uniform(0, np.pi)
        # Random distance from origin (inside the trajectory circle)
        r_lm = rng.uniform(1.0, radius * 0.6)
        lm_a = np.array([r_lm * np.cos(angle), r_lm * np.sin(angle)])
        lm_b = np.array([-r_lm * np.cos(angle), -r_lm * np.sin(angle)])
        idx_a, idx_b = len(lms), len(lms) + 1
        lms.extend([lm_a, lm_b])
        pairs.append((idx_a, idx_b))
    return np.array(lms), pairs


# ------------------------------------------------------------------ #
# Range measurements                                                  #
# ------------------------------------------------------------------ #


def generate_range_measurements(
    poses,
    landmarks,
    sigma_r=0.1,
    assoc_noise=0.0,
    seed=2,
):
    """
    Generate range measurements with optional wrong-association corruption.

    assoc_noise : probability that a measurement is re-attributed to a
                  randomly chosen WRONG landmark.  This creates geometrically
                  irreconcilable factors (outliers), NOT multimodality.
    """
    rng = np.random.default_rng(seed)
    T, K = len(poses), len(landmarks)
    meas = []
    for t in range(T):
        for k in range(K):
            dx = landmarks[k, 0] - poses[t, 0]
            dy = landmarks[k, 1] - poses[t, 1]
            true_range = np.hypot(dx, dy)
            noisy_range = true_range + rng.normal(0, sigma_r)

            reported_k = k
            if assoc_noise > 0 and rng.random() < assoc_noise:
                candidates = [j for j in range(K) if j != k]
                if candidates:
                    reported_k = rng.choice(candidates)

            meas.append((t, reported_k, float(noisy_range)))
    return meas


# ------------------------------------------------------------------ #
# Scenario builders                                                   #
# ------------------------------------------------------------------ #


def build_scenario(
    T=30,
    K=6,
    radius=5.0,
    lm_spread=4.0,
    sigma_range=0.1,
    assoc_noise=0.0,
    seed=0,
):
    """
    Baseline scenario: random landmark layout + optional wrong-association noise.
    Use this to study outlier robustness.
    """
    rng = np.random.default_rng(seed)
    poses = generate_circular_trajectory(T, radius, seed)
    landmarks = rng.uniform(-lm_spread, lm_spread, (K, 2))
    odom = add_odometry_noise(poses, seed=seed + 1)
    meas = generate_range_measurements(
        poses, landmarks, sigma_range, assoc_noise, seed=seed + 2
    )
    return SLAMScenario(poses, landmarks, odom, meas, assoc_noise, sigma_range)


def build_loop_closure_scenario(
    T=40,
    sigma_range=0.1,
    odom_drift=0.3,
    seed=0,
):
    """
    Loop closure ambiguity scenario.

    The robot travels a figure-8 / lollipop path that brings it back near
    its starting position after accumulating significant odometry drift.
    When the loop-closure range measurement fires, two hypotheses are
    equally consistent:
      A) robot is at the true return position  (correct)
      B) robot is offset by the accumulated drift (iSAM2 MAP)

    This creates a bimodal pose posterior that Gaussian solvers collapse
    onto one mode of.  The condition number of the joint pose+landmark
    covariance spikes at the loop-closure window because the new constraint
    is nearly rank-deficient given the drifted prior.

    Parameters
    ----------
    T          : total timesteps
    sigma_range: range measurement noise std
    odom_drift : std of per-step odometry noise (high → more drift → stronger ambiguity)
    seed       : random seed
    """
    rng = np.random.default_rng(seed)

    # --- ground truth: out-and-back loop
    # First half: straight line out
    # Second half: arc returning near start
    half = T // 2
    poses = np.zeros((T, 3))
    for t in range(1, half):
        poses[t] = poses[t - 1] + np.array([0.5, 0.0, 0.0])
    # turn and come back
    for t in range(half, T):
        angle = np.pi * (t - half) / (T - half)
        poses[t, 0] = poses[half - 1, 0] - 0.5 * (t - half) * np.cos(angle)
        poses[t, 1] = 0.5 * np.sin(angle) * (t - half) * 0.3
        poses[t, 2] = angle

    # --- landmarks: one near start, one near end of outbound leg
    landmarks = np.array(
        [
            [0.0, 1.5],  # L0: near start — loop closure anchor
            [poses[half - 1, 0], 1.5],  # L1: far end
        ]
    )

    # --- noisy odometry (high drift)
    odom = []
    for t in range(T - 1):
        dp = poses[t + 1] - poses[t]
        noise = rng.normal(0, [odom_drift, odom_drift, 0.05])
        odom.append(dp + noise)
    odom = np.array(odom)

    # --- range measurements: all poses observe both landmarks
    meas = []

    # in generate_range_measurements (or inline in build_loop_closure_scenario):
    MAX_RANGE = 4.0  # only observe landmarks within this distance
    for t in range(T):
        for k, lm in enumerate(landmarks):
            r_true = np.linalg.norm(poses[t, :2] - lm)
            if r_true > MAX_RANGE:
                continue  # landmark not visible from this pose
            r_noisy = r_true + rng.normal(0, sigma_range)
            meas.append((t, k, float(r_noisy)))

    return SLAMScenario(poses, landmarks, odom, meas, 0.0, sigma_range)


def build_single_pose_ambiguous():
    """
    One robot pose, two landmarks at equal range.
    Posterior over landmark positions is analytically bimodal.
    ESS should drop sharply; MAP Δlogp should be large.
    """
    pose = np.array([[0.0, 0.0, 0.0]])
    landmarks = np.array([[3.0, 1.0], [3.0, -1.0]])  # symmetric about x-axis
    # one range measurement to each: both measure r=sqrt(10)
    r = np.sqrt(10)
    meas = [(0, 0, r), (0, 1, r)]
    odom = np.zeros((0, 3))
    return SLAMScenario(pose, landmarks, odom, meas, 0.0, 0.1)