## High severity

- [ ]  Validate the ellipsoid-normal gravity surrogate under anisotropic soft iron

  - **Summary:** The convex gravity surrogate is exact for the ellipsoid normal, but only approximates constant
    magnetic dip when the soft-iron correction is anisotropic, so a large gravity weight can bias the fit.
  - **Position:** `src/fusion/mag_calibrator.rs (gravity_surrogate_anisotropy)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_gravity_surrogate_survives_strong_anisotropy)`

## Medium severity

- [ ]  Score replacement candidates in their post-replacement buffer

  - **Summary:** Candidate and victim diversity scores currently use different neighbor pools.
  - **Position:** `src/fusion/mag_calibrator.rs (candidate_score_includes_replaced_victim)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_candidate_score_includes_replaced_victim)`
