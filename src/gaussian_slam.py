import gtsam
from gtsam import symbol_shorthand as sb
import numpy as np

X = sb.X  # robot pose symbols
L = sb.L  # landmark symbols

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.01]))
ODOM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))


def build_range_noise(sigma):
    return gtsam.noiseModel.Isotropic.Sigma(1, sigma)


class GaussianSLAM:
    def __init__(self, sigma_range=0.1):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self.sigma_range = sigma_range
        self.range_noise = build_range_noise(sigma_range)
        self._initialised_lm = set()
        self._lm_obs_count = {}

    def initialise(self, first_pose: np.ndarray):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose0 = gtsam.Pose2(*first_pose)
        graph.push_back(gtsam.PriorFactorPose2(X(0), pose0, PRIOR_NOISE))
        values.insert(X(0), pose0)
        self.isam.update(graph, values)
        self._prev_pose = pose0

    def step(
        self, t: int, odom: np.ndarray, range_meas_at_t: list, lm_init_guess: np.ndarray
    ):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        rel = gtsam.Pose2(*odom)
        graph.push_back(gtsam.BetweenFactorPose2(X(t - 1), X(t), rel, ODOM_NOISE))

        prev = self.isam.calculateEstimate().atPose2(X(t - 1))
        new_pose = prev.compose(rel)
        values.insert(X(t), new_pose)

        # count observations before deciding to add factor
        for k, r_meas in range_meas_at_t:
            self._lm_obs_count[k] = self._lm_obs_count.get(k, 0) + 1

            if self._lm_obs_count[k] == 3:
                lm_pt = gtsam.Point2(*lm_init_guess[k])
                values.insert(L(k), lm_pt)
                self._initialised_lm.add(k)
                # weak prior to keep landmark constrained
                lm_noise = gtsam.noiseModel.Isotropic.Sigma(2, 5.0)
                graph.push_back(gtsam.PriorFactorPoint2(L(k), lm_pt, lm_noise))

            if self._lm_obs_count[k] >= 3:
                graph.push_back(
                    gtsam.RangeFactor2D(X(t), L(k), float(r_meas), self.range_noise)
                )

        self.isam.update(graph, values)

    def get_estimate(self):
        return self.isam.calculateEstimate()

    def get_marginals(self):
        est = self.get_estimate()
        marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), est)
        return marginals, est
