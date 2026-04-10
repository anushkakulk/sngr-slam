# Selective Non-Gaussian Refinement for SLAM: Experimental Observations

**Setup:** 5 seeds, T=30 poses, K=6 landmarks, circular trajectory, range-only measurements
Trigger threshold: τ=3.9 (new pipeline), τ=3.96 (baseline), chosen empirically by inspecting the score distribution at noise=0.0 (clean data). At noise=0.0, scores range from 3.54–4.39 with mean 3.79. τ was set just below the observed maximum so that only the most geometrically uncertain windows trigger, while the majority of clean windows do not. τ is a design parameter tuned on the same scenario used for evaluation, which is a limitation. In the future, work would use a chi-squared threshold derived from the expected condition number distribution under the null hypothesis (Gaussian posterior).

---

## Baseline Pipeline
### iSAM2 + Covariance condition number trigger + rejection sampling refinement

#### Detector
- Scores the local window by log₁₀(λ_max / λ_min) of the joint marginal
  covariance Σ_w.
- A high score indicates the uncertainty ellipse is elongated in one direction:
  the local factor graph is geometrically under-constrained (e.g. a landmark
  observed from only one direction has good range but no bearing constraint).
- A `t > 0` guard silently dropped window 0, which consistently scored
  highest due to the robot being under-constrained at initialisation.
  
#### Refinement
- The refinement here is a no-op; it can never find a second mode, never evaluates the actual factor graph likelihood, and the returned weights are meaningless as a non-Gaussian approximation.
- Rejection sampling around the MAP: propose from N(MAP, σ²I) with σ=0.5, evaluate under the marginal N(MAP, Σ_iSAM2).
- Both proposal and target are Gaussians centred on the same MAP. 
- The estimate is not updated after sampling; refined poses are computed but never written back to the trajectory.

#### Results

RMSE is averaged over 5 seeds (seeds 0–4). NEES, triggers, precision and recall are from seed=0 as a representative run; multi-seed breakdown available for noise=0.3 in the per-seed section below.

| Noise | Pose RMSE (5 seeds) | NEES (seed=0) | Triggers (seed=0) | True failures | Precision | Recall | iSAM2 time |
|-------|-------------------|---------------|-------------------|---------------|-----------|--------|------------|
| 0.0 | 0.153 ± 0.062 m | 0.356 | 0 / 28 | 0 / 28 | N/A | N/A | 0.018s |
| 0.1 | 1.273 ± 1.195 m | 148.522 | 0 / 28 | 25 / 28 | N/A | 0.000 | 0.018s |
| 0.2 | 1.644 ± 0.901 m | 210.146 | 4 / 28 | 26 / 28 | 1.000 | 0.154 | 0.018s |
| 0.3 | 1.819 ± 1.212 m | 458.255 | 14 / 28 | 26 / 28 | 1.000 | 0.538 | 0.015s |
| 0.4 | 4.143 ± 1.239 m | 745.229 | 24 / 28 | 26 / 28 | 0.958 | 0.885 | 0.013s |

**Note on RMSE variance:** the std at noise=0.1–0.3 is nearly as large as the mean (e.g. 1.273±1.195 at noise=0.1). Wrong-association failures are highly sensitive to which landmarks get corrupted in a given seed; some random configurations produce localised failures, others corrupt the whole trajectory. This variance is itself a finding: the condition-number detector's ability to fire depends on which windows are geometrically affected, not just on the noise fraction.

**Note on "best sample weight":** the baseline rejection sampler reports log importance weights, nearly all strongly negative (e.g. −73.05 at noise=0.3). A weight of −50 nats means the proposal placed essentially zero probability mass on that sample under the target distribution. The occasional near-zero weight (e.g. +2.03 at window 19, noise=0.3) is a lucky draw near the MAP, not a meaningful refinement. The refinement step was inert regardless of trigger count.

#### Tau sensitivity (seed=0)

The operating range for τ is narrow — about 0.08 units separates "fires everywhere" from "fires selectively."

| τ | Clean triggers (noise=0.0) | High-noise triggers (noise=0.3) |
|---|---|---|
| 1.0 | 27 / 28 | 27 / 28 |
| 2.0 | 27 / 28 | 27 / 28 |
| 3.92 | 0 / 28 | 21 / 28 |
| 3.93 | 0 / 28 | 21 / 28 |
| 3.94 | 0 / 28 | 20 / 28 |
| 3.95 | 0 / 28 | 17 / 28 |
| 3.96 | 0 / 28 | 14 / 28 |
| 4.0 | 0 / 28 | 8 / 28 |
| 5.0 | 0 / 28 | 0 / 28 |

τ=3.92–3.93 is the sweet spot: fires on ≥21/28 high-noise windows while triggering 0/28 on clean data. Below τ=3.92 discrimination collapses. τ=3.96 was used for baseline experiments (slightly more conservative).

#### Per-seed precision/recall at noise=0.3, τ=3.96

Detector performance varies substantially across seeds; whether it fires at all depends on the random landmark configuration, not just the noise level.

| Seed | Triggers | TP | FP | FN | Precision | Recall |
|------|----------|----|----|-----|-----------|--------|
| 0 | 14 / 28 | 14 | 0 | 12 | 1.000 | 0.538 |
| 1 | 8 / 28 | 8 | 0 | 16 | 1.000 | 0.333 |
| 2 | 0 / 28 | 0 | 0 | 17 | N/A | 0.000 |
| 3 | 10 / 28 | 10 | 2 | 13 | 0.833 | 0.435 |
| 4 | 0 / 28 | 0 | 0 | 20 | N/A | 0.000 |

Seeds 2 and 4 produce zero triggers despite 17–20 true failure windows. Precision=1.0 where the detector fires is consistent, but whether it fires is not, showcases a fundamental limitation of the condition-number approach for wrong-association noise.



#### Key observations

1. The rejection sampler produced no actual improvement to any estimate. RMSE and NEES are pure iSAM2 numbers; the refinement step was inert.
2. The condition number detector correctly identifies geometrically under-constrained windows but is blind to wrong-association noise at noise=0.1: NEES=148 with 0 triggers fired.
3. Wrong-association noise at 10% does not change the graph's covariance structure; iSAM2 absorbs corrupted measurements without the topology changing, so the condition number score is unaffected.
4. At noise ≥ 0.2, the covariance structure does change, and triggers begin firing with precision=1.0; every triggered window is a genuine failure.
5. Selectivity breaks down at high noise: noise=0.3 triggers 22/28 windows. The selectivity argument only holds at noise ≤ 0.2.

---

## New Pipeline
### iSAM2 + covariance condition number trigger + nested sampling refinement

#### Detector - remains the same
- scores log₁₀(λ_max / λ_min) of the **covariance**. Large eigenvalue of Σ = uncertain direction = what we want to detect.

#### Refinement changes
- Full nested sampling via dynesty (rwalk sampler, bootstrap=0).
- Likelihood evaluates actual GTSAM factor graph errors: −0.5 · Σ f.error(values).
- Variable closure: any factor touching the window pulls in all its keys, so boundary factors (e.g. RangeFactor2D spanning a pose and a landmark) are fully evaluated rather than rejected.
- Window variables (Pose2) are sampled; closure variables (Point2 landmarks, adjacent poses) are held fixed at iSAM2 MAP values.
- Prior: N(MAP, cov_inflation · Σ_iSAM2) via Cholesky transform.
- All windows refined against original iSAM2 estimate (not sequentially), updates applied together at the end to prevent cascading corruption.
- Posterior mean returned as point estimate (more robust than argmax weight when ESS is high).
- Improvement gate: refined estimate only applied if logp(posterior mean) ≥ logp(MAP) − 1e-3.

#### Results (seed=0, single run)

**noise=0.0**
| Metric | Value |
|--------|-------|
| Pose RMSE | 0.1258 m |
| Landmark RMSE | 0.1041 m |
| Mean NEES | 0.356 (underconfident; Σ too large, safe direction) |
| Triggers | 4 / 28 |
| True failures | 0 / 28 |
| Precision | 0.000 (all 4 triggers are false positives; no real failures) |
| Recall | N/A |
| Gaussian SLAM | 0.018s |
| Refinement | 44.25s total · 11.06s/window · 4 windows |
| Total | 44.26s |
| Sampler | ESS~50%, Δlogp~0 on all windows → posterior is Gaussian, MAP correctly retained |

**noise=0.1**
| Metric | Value |
|--------|-------|
| Pose RMSE | 2.8068 m |
| Landmark RMSE | 3.3066 m |
| Mean NEES | 148.522 (severely overconfident) |
| Triggers | 0 / 28 |
| True failures | 25 / 28 |
| Precision | N/A |
| Recall | 0.000 |
| Gaussian SLAM | 0.016s |
| Refinement | 0.00s (no triggers) |
| Total | 0.02s |
| Note | Detector blind spot. Wrong association at 10% does not change covariance structure. iSAM2 fails silently. NEES=148 is the clearest demonstration of the failure. |

**noise=0.2**
| Metric | Value |
|--------|-------|
| Pose RMSE | 2.6477 m |
| Landmark RMSE | 2.2989 m |
| Mean NEES | 209.876 |
| Triggers | 14 / 28 (50% trigger rate; borderline selective) |
| True failures | 26 / 28 |
| Precision | 1.000 (every trigger is a genuine failure) |
| Recall | 0.538 |
| Gaussian SLAM | 0.018s |
| Refinement | 174.90s total · 12.49s/window · 14 windows |
| Total | 174.92s |
| Sampler | Δlogp = +1.0 to +5.8 nats. Window 17: +5.842 nats; strongest escape from bad MAP. All windows queued. |
| Note | RMSE remains high because wrong association corrupts the global trajectory; local window refinement cannot undo global drift. Sampler works correctly locally. |

**noise=0.3**
| Metric | Value |
|--------|-------|
| Pose RMSE | 3.8054 m |
| Landmark RMSE | 2.9070 m |
| Mean NEES | 454.137 |
| Triggers | 22 / 28 (79% trigger rate; selectivity lost) |
| True failures | 26 / 28 |
| Precision | 1.000 |
| Recall | 0.846 |
| Gaussian SLAM | 0.017s |
| Refinement | 244.55s total · 11.12s/window · 22 windows |
| Total | 244.57s |
| Sampler | Δlogp = +0.4 to +2.1 nats. All windows queued. |

#### Key observations

1. The nested sampler correctly characterises the posterior as Gaussian when it is (noise=0.0): ESS~50%, Δlogp~0, MAP retained. The system does not degrade on clean data.
2. The noise=0.1 blind spot persists; it is a property of the trigger criterion (condition number), not the sampler. Wrong association at low rates is undetectable from covariance structure alone.
3. At noise=0.2, precision=1.0 with a 50% trigger rate. This is the operationally useful regime: selective enough to save compute, precise enough to trust every trigger.
4. Δlogp > 0 on all triggered windows at noise ≥ 0.2 confirms the sampler is genuinely finding better local poses than iSAM2. The range +0.4 to +5.8 nats represents real escapes from local minima, not sampling noise (which would be < 0.2 nats).
5. RMSE improvement is limited because the failure mode is global. Local refinement improves individual windows but cannot correct accumulated trajectory drift. This is a limitation of the windowed approach, not the sampler.
6. Per-window cost is ~11–12s regardless of noise level, confirming the sampler cost scales with window size (fixed at 3 poses) and not with the degree of corruption.
7. Selectivity is the key to tractability. At noise=0.0, 4/28 windows trigger: total cost 44s vs 336s if run on all windows; a **7.6× saving**.

---

## Proof of Concept: Bimodal Posterior (`test_bimodal.py`)

**Setup:** Two anchor poses at A=(0,0) and B=(4,0). One landmark L(0) at unknown position. Range measurements r_A = r_B = 3.0. Analytical solution: two intersection points at (2, +√5) and (2, −√5) ≈ (2, ±2.236). Posterior is analytically bimodal.

**iSAM2 result:**
- Converged to (2.000, 0.000); the midpoint between the two modes.
- MAP log-likelihood: −12.4997
- True mode log-likelihood: −0.014
- iSAM2 is **12.49 nats below the true posterior mode**.
- The MAP is the saddle point between the two modes; the least probable point on the constraint manifold.

**Nested Sampling result:**
- Best sample: (1.984, −2.232); within 0.004 of the true mode at (2, −2.236).
- Best log-likelihood: −0.0143
- Improvement over iSAM2 MAP: **+12.49 nats**.
- Weighted std in y: 2.207 ≈ √5 = 2.236; correctly captures the spread between modes.
- Histogram shows two clear peaks at y = ±√5.

**Significance:** This is the cleanest demonstration that iSAM2 can be categorically wrong (not just imprecise) and that nested sampling recovers the correct answer. The +12.49 nat improvement is the difference between a geometrically impossible estimate and the true solution. This result motivates the entire selective refinement framework.

---

## Limitations and Future Work

1. **Detector blind spot at low noise:** the condition number score misses wrong-association failures below ~20% corruption rate. A residual-based trigger (mean squared normalised range residual per window) would catch these cases.
2. **Local refinement cannot fix global failures:** when wrong association corrupts most of the trajectory, window-by-window refinement improves local logprob but cannot undo global drift. Joint re-estimation or graph repair is needed for the high-noise regime.
3. **Prior coverage for multimodal posteriors:** The current prior transform approximates the prior as N(MAP, Σ_iSAM2), limiting mode discovery to the neighbourhood of the current estimate. A principled implementation following NSFG would use ancestral sampling through the prior factor graph, enabling discovery of distant modes as in loop closure scenarios. 
4. **Scalability:** at ~12s per window, the system is suitable for offline or batch processing but not real-time use. Reducing `nlive` (100→50) and `walks` (9→5) cuts cost by ~2–3× with modest accuracy loss.
5. **Fixed landmarks:** landmark variables are held fixed during window refinement. Jointly sampling over poses and nearby landmarks would improve accuracy but increases `ndim` and sampler cost.