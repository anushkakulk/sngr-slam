import numpy as np
import time
import gtsam

from data_gen import build_scenario, build_loop_closure_scenario
from gaussian_slam import GaussianSLAM, X, L
from covariance_condition_number_detector import score_all_windows
from nested_sampling_refinement import (
    refine_window,
    refine_window_with_landmarks,
    LocalFactorGraphLikelihood,
    build_closure,
    extract_local_factors,
    _build_mean,
    _LandmarkLikelihood,
)
from evaluate import pose_rmse, landmark_rmse, nees, trigger_precision_recall


def poorly_constrained_landmarks(
    marginals, lm_keys, win_keys, estimate, variance_threshold=2.0
):
    """
    Return landmark keys observed by the window that have high marginal
    variance — candidates to sample jointly with the poses.
    """
    candidates = []
    for lk in lm_keys:
        if not estimate.exists(lk):
            continue
        try:
            cov = marginals.marginalCovariance(lk)
            if np.trace(cov) > variance_threshold:
                candidates.append(lk)
        except Exception:
            pass
    return candidates


def find_true_failure_windows(estimate, ground_truth_poses, window_size=3, epsilon=0.5):
    """
    A window is a true failure if the mean pose error within it exceeds epsilon.
    Used as ground truth for trigger precision/recall.
    """
    T = len(ground_truth_poses)
    failure_windows = []
    for t in range(T - window_size + 1):
        errs = []
        for tt in range(t, t + window_size):
            ep = estimate.atPose2(X(tt))
            est_xy = np.array([ep.x(), ep.y()])
            true_xy = ground_truth_poses[tt, :2]
            errs.append(np.linalg.norm(est_xy - true_xy))
        if np.mean(errs) > epsilon:
            failure_windows.append(t)
    return failure_windows


def run(
    assoc_noise=0.0,
    tau=5.0,
    seed=0,
    verbose=True,
    ambiguous=False,
    nlive=100,
    maxiter=1000,
    odom_drift=0.3,
    cov_inflation=1.0,
):
    """
    Run the full SNGR SLAM pipeline on a generated scenario, with options to induce different types of failures (association noise or loop closure ambiguity).
    Parameters
    ----------
    assoc_noise : float
        Std-dev of Gaussian noise added to landmark range measurements (m).
        0.0 = perfect measurements; >=0.2 starts triggering refinement windows.
        Higher values introduce more linearization error in iSAM2.
    tau : float
        Condition-number threshold for the covariance detector. Lower = more
        sensitive (more windows flagged); higher = only severe cases flagged.
        Empirically tuned sweet spot is 3.9–5.0 (TODO: automate tuning).
    seed : int
        Random seed for scenario generation (poses, landmarks, noise draws).
        Change to verify results generalise across scenarios.
    verbose : bool
        If True, prints per-window Δlogp and ESS to stdout during refinement.
    ambiguous : bool
        If True, places landmarks so multiple associations are plausible,
        stress-testing loop-closure ambiguity handling.
    nlive : int
        Number of live points in the nested sampler. Higher = better posterior
        coverage and ESS at the cost of runtime. Minimum 100 for 9-D windows;
    maxiter : int
        Maximum nested sampling iterations before forced termination. Caps
        runtime per window; too low causes early exit and underestimated Δlogp.
    odom_drift : float
        Std-dev of cumulative odometry drift injected into dead-reckoning (m).
        Higher values weaken the pose prior and produce more triggered windows.
        Set to 0.0 for a drift-free baseline.
    cov_inflation : float
        Scalar multiplier applied to all iSAM2 marginal covariances before
        being passed to the detector and nested sampler prior. Values >1.0
        widen the sampling prior (safer but slower); 1.0 = no inflation.
    """
    if ambiguous:
        scenario = build_loop_closure_scenario(
            T=40, sigma_range=0.1, odom_drift=odom_drift, seed=seed
        )
    else:
        scenario = build_scenario(T=30, K=6, assoc_noise=assoc_noise, seed=seed)

    T = len(scenario.poses)

    # normal gaussian SLAM
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

    pose_keys = [X(t) for t in range(T)]
    lm_keys = [L(k) for k in range(len(scenario.landmarks))]

    # trigger detection
    scores, triggers = score_all_windows(
        marginals, pose_keys, lm_keys, tau=tau, meas_by_t=meas_by_t
    )
    triggers = list(dict.fromkeys(triggers))  # deduplicate, preserve order

    score_vals = [s for _, s in scores]
    if verbose:
        print(
            f"Score range: min={min(score_vals):.3f}  "
            f"max={max(score_vals):.3f}  mean={np.mean(score_vals):.3f}"
        )

    # build refined_estimate (poses + landmarks)
    refined_estimate = gtsam.Values()
    for t in range(T):
        refined_estimate.insert(X(t), estimate.atPose2(X(t)))
    for k in range(len(scenario.landmarks)):
        lk = L(k)
        if estimate.exists(lk):
            refined_estimate.insert(lk, estimate.atPoint2(lk))

    # refinement
    # All windows are refined against the original iSAM2 estimate so that
    # refinement of window N does not corrupt the prior for window N+1.
    # Updates are collected and applied together at the end.
    refinement_times = []
    pending_updates = {}  # win_start -> (win_keys, refined_state)

    for win_start in triggers:
        win_keys = pose_keys[win_start : win_start + 3]

        # find poorly constrained landmarks in this window
        lm_to_sample = poorly_constrained_landmarks(
            marginals,
            lm_keys,
            win_keys,
            refined_estimate,
            variance_threshold=2.0,
        )

        t_ref_start = time.time()
        if lm_to_sample:
            if verbose:
                names = [gtsam.Symbol(k).string() for k in lm_to_sample]
                print(f"  → sampling {len(lm_to_sample)} landmarks jointly: {names}")
            refined_state, weights, all_samples = refine_window_with_landmarks(
                slam,
                refined_estimate,
                marginals,
                win_keys,
                lm_keys_to_sample=lm_to_sample,
                nlive=nlive,
                maxiter=maxiter,
                verbose=verbose,
            )
            # for the improvement check, use _LandmarkLikelihood
            closure_keys = build_closure(slam.isam, win_keys + lm_to_sample)
            factors = extract_local_factors(slam.isam, closure_keys)
            _lhood = _LandmarkLikelihood(
                factors, win_keys, lm_to_sample, closure_keys, refined_estimate
            )
            map_vec = _build_mean(refined_estimate, win_keys, lm_to_sample)
        else:
            refined_state, weights, all_samples = refine_window(
                slam,
                refined_estimate,
                marginals,
                win_keys,
                nlive=nlive,
                maxiter=maxiter,
                cov_inflation=cov_inflation,
                verbose=verbose,
            )
        t_ref = time.time() - t_ref_start
        refinement_times.append(t_ref)

        # check improvement against the original estimate
        closure_keys = build_closure(slam.isam, win_keys)
        factors = extract_local_factors(slam.isam, closure_keys)
        _lhood = LocalFactorGraphLikelihood(
            factors, win_keys, closure_keys, refined_estimate
        )
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
        map_logp = _lhood(map_vec)
        refined_logp = _lhood(refined_state)

        improved = refined_logp >= map_logp - 1e-3
        if improved:
            pending_updates[win_start] = (
                win_keys,
                refined_state,
                lm_to_sample if lm_to_sample else [],
            )

        if verbose:
            print(
                f"  Window {win_start}: {t_ref:.2f}s | {len(all_samples)} samples | "
                f"Δlogp={refined_logp - map_logp:+.3f} | "
                f"{'queued' if improved else 'kept MAP'}"
            )

    # apply all accepted refinements at once
    for win_start, (win_keys, refined_state, lm_sampled) in pending_updates.items():
        for i, k in enumerate(win_keys):
            idx = 3 * i
            pose = gtsam.Pose2(*refined_state[idx : idx + 3])
            refined_estimate.update(k, pose)
        # update landmark positions if they were sampled
        offset = 3 * len(win_keys)
        for i, lk in enumerate(lm_sampled):
            idx = offset + 2 * i
            pt = gtsam.Point2(refined_state[idx], refined_state[idx + 1])
            if refined_estimate.exists(lk):
                refined_estimate.erase(lk)
            refined_estimate.insert(lk, pt)

    # evaluation: pose RMSE
    est_poses = np.array(
        [refined_estimate.atPose2(X(t)).translation() for t in range(T)]
    )
    rmse = pose_rmse(
        np.hstack([est_poses, np.zeros((T, 1))]),
        scenario.poses,
    )

    # evaluation: NEES (2-D x,y, consistent ≈ 2.0)
    nees_vals = []
    for t in range(T):
        try:
            cov_full = marginals.marginalCovariance(X(t))
            cov_xy = cov_full[:2, :2]
            est_xy = refined_estimate.atPose2(X(t)).translation()
            gt_xy = scenario.poses[t, :2]
            nees_vals.append(nees(est_xy, gt_xy, cov_xy))
        except Exception:
            pass
    mean_nees = float(np.mean(nees_vals)) if nees_vals else float("nan")

    # evaluation: landmark RMSE
    lm_est, lm_gt = [], []
    for k in range(len(scenario.landmarks)):
        if refined_estimate.exists(L(k)):
            lm_est.append(refined_estimate.atPoint2(L(k)))
            lm_gt.append(scenario.landmarks[k])
    lm_rmse = (
        landmark_rmse(np.array(lm_est), np.array(lm_gt)) if lm_est else float("nan")
    )

    # evaluation: trigger precision/recall
    true_failures = find_true_failure_windows(estimate, scenario.poses)
    pr = trigger_precision_recall(triggers, true_failures, len(scores))

    total_refinement = sum(refinement_times)

    if verbose:
        label = f"drift={odom_drift}" if ambiguous else f"noise={assoc_noise}"
        print(f"\n{label}")
        print(f"Pose RMSE:      {rmse:.4f} m")
        print(f"Landmark RMSE:  {lm_rmse:.4f} m")
        print(f"Mean NEES:      {mean_nees:.3f}  (consistent ≈ 2.0)")
        print(f"Triggers:       {len(triggers)} / {len(scores)}")
        print(f"True failures:  {len(true_failures)} / {len(scores)}")
        prec_str = f"{pr['precision']:.3f}" if pr["precision"] is not None else "N/A"
        rec_str = f"{pr['recall']:.3f}" if pr["recall"] is not None else "N/A"
        print(f"Precision:      {prec_str}")
        print(f"Recall:         {rec_str}")
        print(f"Gaussian SLAM:  {gaussian_time:.3f}s")
        if refinement_times:
            print(
                f"Refinement:     {total_refinement:.2f}s total | "
                f"{np.mean(refinement_times):.2f}s/window | "
                f"{len(triggers)} windows"
            )
        else:
            print(f"Refinement:     0.00s (no triggers)")
        print(f"Total:          {gaussian_time + total_refinement:.2f}s")

    return rmse, mean_nees, lm_rmse, gaussian_time, total_refinement, pr


if __name__ == "__main__":
    print("=== Experiment 1: wrong data association (outlier noise) ===")
    for noise in [0.0, 0.1, 0.2, 0.3]:
        run(assoc_noise=noise, tau=3.9, verbose=True)

    print("\n=== Experiment 2: loop closure ambiguity ===")
    for drift in [0.3, 0.5, 0.8]:
        run(
            tau=3.5,
            verbose=True,
            ambiguous=True,
            nlive=200,
            maxiter=3000,
            seed=0,
            odom_drift=drift,
            cov_inflation=25.0,
        )
