"""
Computes the local ambiguity score Aw = log10(kappa(H_w))
where kappa is the condition number of the local Hessian block.
"""

import numpy as np
import gtsam


def extract_local_hessian(marginals, symbols: list) -> np.ndarray:
    keys = gtsam.KeyVector()
    for s in symbols:
        keys.append(s)
    cov = marginals.jointMarginalCovariance(keys).fullMatrix()
    H = np.linalg.inv(cov)
    return H


def ambiguity_score(H: np.ndarray) -> float:
    """
    Aw = log10(kappa(H)) = log10(lambda_max / lambda_min)
    A large score signals that the posterior is flat in some direction
    → Gaussian approximation may be unreliable.
    """
    eigs = np.linalg.eigvalsh(H)
    eigs = np.abs(eigs)
    eigs = eigs[eigs > 1e-10]  # ignore numerically zero eigenvalues
    if len(eigs) < 2:
        return 0.0
    return float(np.log10(eigs.max() / eigs.min()))


def score_all_windows(
    marginals, pose_keys, lm_keys, window_size=3, tau=4.0, meas_by_t=None
):
    """
    Slide a window of `window_size` consecutive poses + their observed
    landmarks, score each window, and flag those above threshold tau.

    Returns:
      scores  : list of (window_start_t, score)
      triggers: list of window_start_t indices that exceeded tau
    """
    T = len(pose_keys)
    scores, triggers = [], []
    for t in range(T - window_size + 1):
        win_poses = pose_keys[t : t + window_size]
        win_lm_indices = set()
        if meas_by_t is not None:
            for tt in range(t, t + window_size):
                for k, r in meas_by_t.get(tt, []):
                    win_lm_indices.add(k)
        win_lms = [lm_keys[k] for k in win_lm_indices if k < len(lm_keys)]
        symbols = win_poses + win_lms
        try:
            H = extract_local_hessian(marginals, symbols)
            s = ambiguity_score(H)
        except Exception:
            s = 0.0
        scores.append((t, s))
        # skip window 0 — always underconstrained at start
        if s > tau and t > 0:
            triggers.append(t)
    return scores, triggers
