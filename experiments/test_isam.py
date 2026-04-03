import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_gen import build_scenario
from gaussian_slam import GaussianSLAM, X, L
import numpy as np

s = build_scenario(T=15, K=4, assoc_noise=0.0, seed=0)

slam = GaussianSLAM(sigma_range=s.range_noise)
slam.initialise(s.poses[0])

meas_by_t = {t: [] for t in range(len(s.poses))}
for t, k, r in s.range_meas:
    meas_by_t[t].append((k, r))

for t in range(1, len(s.poses)):
    slam.step(t, s.odometry[t - 1], meas_by_t[t], s.landmarks)

est = slam.get_estimate()
print("iSAM2 ran successfully")
print("Pose 0 estimate:", est.atPose2(X(0)))
print("Pose 1 estimate:", est.atPose2(X(1)))
