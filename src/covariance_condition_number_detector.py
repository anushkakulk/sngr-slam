import numpy as np
import gtsam


def extract_local_hessian(marginals, symbols):
    keys = gtsam.KeyVector()
    for s in symbols:
        keys.append(s)

    cov = marginals.jointMarginalCovariance(keys).fullMatrix()
    return cov


def ambiguity_score(cov):
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.abs(eigs)
    eigs = eigs[eigs > 1e-10]  # ignore near-zero (well-constrained) directions
    if len(eigs) < 2:
        return 0.0
    return float(np.log10(eigs.max() / eigs.min()))


def score_all_windows(
    marginals,
    pose_keys,
    lm_keys,
    window_size=3,
    tau=2.0,
    meas_by_t=None,
):
    T = len(pose_keys)
    scores, triggers = [], []

    for t in range(T - window_size + 1):
        win_poses = pose_keys[t : t + window_size]

        win_lm_indices = set()
        if meas_by_t is not None:
            for tt in range(t, t + window_size):
                for k, _ in meas_by_t.get(tt, []):
                    win_lm_indices.add(k)

        win_lms = [lm_keys[k] for k in win_lm_indices if k < len(lm_keys)]
        symbols = win_poses + win_lms

        try:
            cov = extract_local_hessian(marginals, symbols)
            s = ambiguity_score(cov)
        except Exception:
            s = 0.0

        scores.append((t, s))

        # instead of t > 0, skip only if the window has too few measurements
        if s > tau and t >= window_size - 1:
            triggers.append(t)

    return scores, triggers
