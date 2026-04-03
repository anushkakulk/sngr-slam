import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SLAMScenario:
    poses: np.ndarray  # (T, 3) ground-truth [x, y, theta]
    landmarks: np.ndarray  # (K, 2) ground-truth [x, y]
    odometry: np.ndarray  # (T-1, 3) [dx, dy, dtheta] + noise
    range_meas: List[Tuple]  # (t, k, range) — possibly ambiguous
    # each entry is (timestep, true_landmark_idx, measured_range)
    assoc_noise: float = 0.0  # fraction of measurements with wrong assoc
    range_noise: float = 0.1  # std of range measurement noise


def generate_circular_trajectory(T=30, radius=5.0, seed=0):
    angles = np.linspace(0, 2 * np.pi, T, endpoint=False)
    poses = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), angles + np.pi / 2]
    )
    return poses


def add_odometry_noise(poses, sigma_t=0.05, sigma_r=0.01, seed=1):
    """
    Add noise to odometry measurements derived from the true poses.
    Odometry is in the robot's local frame, so we transform the world-frame deltas accordingly.
    """
    rng = np.random.default_rng(seed)
    T = len(poses)
    odom = []
    for i in range(T - 1):
        # transform world-frame delta into robot-local frame
        dx_world = poses[i + 1, 0] - poses[i, 0]
        dy_world = poses[i + 1, 1] - poses[i, 1]
        theta = poses[i, 2]
        dx_local = np.cos(theta) * dx_world + np.sin(theta) * dy_world
        dy_local = -np.sin(theta) * dx_world + np.cos(theta) * dy_world
        dtheta = poses[i + 1, 2] - poses[i, 2]
        noise = rng.normal([0, 0, 0], [sigma_t, sigma_t, sigma_r])
        odom.append([dx_local + noise[0], dy_local + noise[1], dtheta + noise[2]])
    return np.array(odom)


def generate_range_measurements(poses, landmarks, sigma_r=0.1, assoc_noise=0.0, seed=2):
    """
    For each pose, measure range to every visible landmark.
    assoc_noise: probability that a measurement is attributed to a
                 WRONG nearby landmark (simulates ambiguous data assoc).
    """
    rng = np.random.default_rng(seed)
    T, K = len(poses), len(landmarks)
    meas = []
    for t in range(T):
        for k in range(K):
            dx = landmarks[k, 0] - poses[t, 0]
            dy = landmarks[k, 1] - poses[t, 1]
            true_range = np.sqrt(dx**2 + dy**2)
            noisy_range = true_range + rng.normal(0, sigma_r)
            # Simulate ambiguous data association: with probability assoc_noise,
            # attribute this measurement to a randomly chosen wrong landmark.
            # This is the primary experimental manipulation; by controlling the
            # corruption rate we create a continuum of SLAM difficulty and generate
            # ground truth for evaluating the ambiguity detector.
            reported_k = k
            if assoc_noise > 0 and rng.random() < assoc_noise:
                candidates = [j for j in range(K) if j != k]
                if candidates:
                    reported_k = rng.choice(candidates)
            meas.append((t, reported_k, noisy_range))
    return meas


def build_scenario(
    T=30, K=8, radius=5.0, lm_spread=4.0, sigma_range=0.1, assoc_noise=0.0, seed=0
):
    rng = np.random.default_rng(seed)
    poses = generate_circular_trajectory(T, radius, seed)
    landmarks = rng.uniform(-lm_spread, lm_spread, (K, 2))
    odom = add_odometry_noise(poses, seed=seed + 1)
    meas = generate_range_measurements(
        poses, landmarks, sigma_range, assoc_noise, seed=seed + 2
    )
    return SLAMScenario(poses, landmarks, odom, meas, assoc_noise, sigma_range)
