# SNGR-SLAM: Selective Non-Gaussian Refinement for SLAM Factor Graphs

## Overview

Standard Gaussian SLAM solvers like iSAM2 are efficient but fail silently under
ambiguous data association; committing confidently to wrong estimates with no
internal mechanism to detect the error. This project asks: *when does Gaussian
SLAM fail, and can we detect it at runtime?*

We propose **Selective Non-Gaussian Refinement (SNGR)**, a framework that runs
iSAM2 as a baseline and flags local windows of the factor graph where the
Gaussian approximation is likely unreliable. Detection is based on the condition
number of the local information matrix κ(H_w) where a large condition number signals
that the posterior is flat in some directions and may be multi-modal.

## Repository structure

```
sngr-slam/
├── README.md
├── environment.yml
├── src/
│   ├── data_gen.py
│   ├── gaussian_slam.py
│   ├── detector.py
│   ├── evaluate.py
│   ├── refinement.py
│   └── run.py
├── experiments/
│   ├── plot_results.py
│   ├── test_detector_pipeline.py
│   ├── test_isam.py
└── results/
    └── results_full.png
```

## Setup

```bash
conda env create -f environment.yml
conda activate sngr
```

Requires Python 3.11 and GTSAM 4.2.0 (installed via conda-forge).

## Running experiments

**Generate all results and plots:**
```bash
python experiments/plot_results.py
```

**Run validation experiments:**
```bash
python experiments/test_full.py
```

**Quick iSAM2 sanity check:**
```bash
python experiments/test_isam.py
```

## Method

The pipeline has four phases:

**Phase 1 - Data generation:** Synthetic 2D circular trajectory with K
landmarks and range-only measurements. Association noise fraction controls
what fraction of measurements are misattributed to wrong landmarks,
creating a range from easy unimodal to hard multimodal posteriors.

**Phase 2 - Gaussian baseline:** iSAM2 solves the factor graph
incrementally, returning a MAP estimate (μ, Σ). Landmarks are initialized
after their third observation with a weak prior to prevent underconstrained
linear systems.

**Phase 3 - Ambiguity detection:** A sliding window of 3 poses and their
locally observed landmarks is scored by the condition number of the joint
information matrix: `A_w = log10(κ(H_w))`. Windows exceeding threshold
τ = 3.96 are flagged for refinement. Window 0 is excluded as the factor
graph is still being initialized at the first timestep.

**Phase 4 - Refinement:** Flagged windows are passed to a local sampler
(proof-of-concept placeholder). Full non-Gaussian refinement via nested
sampling over association hypotheses is left as future work.

## References

- Huang, Papalia, and Leonard. *Nested Sampling for Non-Gaussian Inference
  in SLAM Factor Graphs.* IEEE RA-L 2022.
- Dellaert and Kaess. *Factor Graphs for Robot Perception.*
  Foundations and Trends in Robotics, 2017.
- Kaess et al. *iSAM2: Incremental Smoothing and Mapping.*
  IJRR, 2012.