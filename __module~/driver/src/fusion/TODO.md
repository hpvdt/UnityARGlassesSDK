## High severity

- [ ]  Validate the ellipsoid-normal gravity surrogate under anisotropic soft iron

  - **Summary:** The convex gravity surrogate is exact for the ellipsoid normal, but only approximates constant
    magnetic dip when the soft-iron correction is anisotropic, so a large gravity weight can bias the fit.
  - **Position:** `src/fusion/mag_calibrator.rs (gravity_surrogate_anisotropy)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_gravity_surrogate_survives_strong_anisotropy)`

## Medium severity

- [ ]  Bound stale optimizer influence after sample expiry

  - **Summary:** Online parameters retain historical gradient influence after a row is replaced or expires, and
    nothing removes that contribution, so `max_sample_lifespan_us` no longer strictly bounds the estimator's
    effective history. The impact is bounded in practice because gradients come only from currently retained rows,
    so stale influence dilutes as updates track the moving optimum; the main residual risk is slower re-tracking
    after the learning rate has annealed to its floor when the true optimum actually shifts (e.g. hard-iron drift).
  - **Position:** `src/fusion/mag_calibrator.rs (online_history_outlives_sample_lifespan)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_online_history_outlives_sample_lifespan)`
- [ ]  Score replacement candidates in their post-replacement buffer

  - **Summary:** Candidate and victim diversity scores currently use different neighbor pools.
  - **Position:** `src/fusion/mag_calibrator.rs (candidate_score_includes_replaced_victim)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_candidate_score_includes_replaced_victim)`
