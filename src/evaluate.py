import numpy as np
import time


def pose_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """RMSE over [x, y] components of all poses."""
    diff = estimates[:, :2] - ground_truth[:, :2]
    return float(np.sqrt(np.mean(diff**2)))


def landmark_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    diff = estimates - ground_truth
    return float(np.sqrt(np.mean(diff**2)))


def nees(estimate: np.ndarray, truth: np.ndarray, cov: np.ndarray) -> float:
    """
    Normalised Estimation Error Squared.
    NEES ~ 1.0 means the filter is consistent (not over/under-confident).
    """
    err = estimate - truth
    return float(err @ np.linalg.inv(cov) @ err)


def trigger_precision_recall(
    predicted_triggers: list, true_failure_windows: list, total_windows: int
):
    """
    predicted_triggers : window indices flagged by detector
    true_failure_windows: windows where iSAM2 RMSE gap > epsilon
    """
    tp = len(set(predicted_triggers) & set(true_failure_windows))
    fp = len(set(predicted_triggers) - set(true_failure_windows))
    fn = len(set(true_failure_windows) - set(predicted_triggers))

    # undefined cases — no triggers and no failures means detector is correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
