import numpy as np
import time

from data_gen import build_scenario
from gaussian_slam import GaussianSLAM, X, L
from covariance_condition_number_detector import score_all_windows
from importance_sampling_refinement import refine_window
from evaluate import pose_rmse, trigger_precision_recall, nees


def find_true_failure_windows(estimate, ground_truth_poses, window_size=3, epsilon=0.5):
    """
    A window is a true failure if the mean pose error within it exceeds epsilon.
    This defines ground truth for trigger precision/recall.
    """
    T = len(ground_truth_poses)
    failure_windows = []
    for t in range(T - window_size + 1):
        window_errors = []
        for tt in range(t, t + window_size):
            ep = estimate.atPose2(X(tt))
            est_xy = np.array([ep.x(), ep.y()])
            true_xy = ground_truth_poses[tt, :2]
            window_errors.append(np.linalg.norm(est_xy - true_xy))
        if np.mean(window_errors) > epsilon:
            failure_windows.append(t)
    return failure_windows


def run(assoc_noise=0.2, tau=4.0, seed=0, verbose=True):
    # generate SLAM scenario with specified noise and seed
    scenario = build_scenario(T=30, K=6, assoc_noise=assoc_noise, seed=seed)
    T = len(scenario.poses)

    # run iSAM2 on the scenario
    slam = GaussianSLAM(sigma_range=scenario.range_noise)
    slam.initialise(scenario.poses[0])

    meas_by_t = {t: [] for t in range(T)}
    for t, k, r in scenario.range_meas:
        meas_by_t[t].append((k, r))

    lm_init = scenario.landmarks.copy()

    t0 = time.time()
    for t in range(1, T):
        slam.step(t, scenario.odometry[t - 1], meas_by_t[t], lm_init)
    gaussian_time = time.time() - t0

    marginals, estimate = slam.get_marginals()

    # score windows using the detector
    pose_keys = [X(t) for t in range(T)]
    lm_keys = [L(k) for k in range(len(scenario.landmarks))]
    scores, triggers = score_all_windows(
        marginals, pose_keys, lm_keys, tau=tau, meas_by_t=meas_by_t
    )

    # refine flagged windows using non-Gaussian sampling
    for win_start in triggers:
        win_keys = pose_keys[win_start : win_start + 3]
        refined, log_w = refine_window(estimate, marginals, win_keys)
        if verbose:
            print(
                f"  Triggered at window {win_start}, "
                f"best sample weight={np.max(log_w):.2f}"
            )

    # evaluate pose RMSE against ground truth
    est_poses = np.array([estimate.atPose2(X(t)).translation() for t in range(T)])
    rmse = pose_rmse(np.hstack([est_poses, np.zeros((T, 1))]), scenario.poses)

    # evaluate NEES against ground truth
    nees_scores = []
    for t in range(T):
        try:
            ep = estimate.atPose2(X(t))
            est_xy = np.array([ep.x(), ep.y()])
            true_xy = scenario.poses[t, :2]
            cov_full = marginals.marginalCovariance(X(t))
            cov_xy = cov_full[:2, :2]
            n = nees(est_xy, true_xy, cov_xy)
            nees_scores.append(n)
        except Exception:
            pass
    mean_nees = float(np.mean(nees_scores)) if nees_scores else 0.0

    # evaluate trigger precision/recall against true failure windows
    true_failures = find_true_failure_windows(estimate, scenario.poses, epsilon=0.5)
    pr = trigger_precision_recall(triggers, true_failures, len(scores))

    if verbose:
        print(f"\nassoc_noise={assoc_noise}, tau={tau}")
        print(f"  Pose RMSE  : {rmse:.4f} m")
        print(f"  Mean NEES  : {mean_nees:.3f} (2.0 = consistent for 2D)")
        print(f"  Triggers   : {len(triggers)} / {len(scores)} windows")
        print(f"  True failures: {len(true_failures)} / {len(scores)} windows")
        prec_str = f"{pr['precision']:.3f}" if pr["precision"] is not None else "N/A"
        rec_str = f"{pr['recall']:.3f}" if pr["recall"] is not None else "N/A"
        print(f"  Precision  : {prec_str}")
        print(f"  Recall     : {rec_str}")
        print(f"  iSAM2 time : {gaussian_time:.3f}s")

    return {
        "rmse": rmse,
        "triggers": triggers,
        "scores": scores,
        "gaussian_time": gaussian_time,
        "nees": mean_nees,
        "precision": pr["precision"] if pr["precision"] is not None else float("nan"),
        "recall": pr["recall"] if pr["recall"] is not None else float("nan"),
        "true_failures": true_failures,
    }


if __name__ == "__main__":
    for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
        run(assoc_noise=noise, tau=3.96, verbose=True)
