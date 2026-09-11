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
- [ ]  Make live fitness statistics cache-derived and lifespan-aware

  - **Summary:** The radial and gravity fitness statistics never expire: their running mean squares keep describing
    removed rows after expiry or replacement (pinned byte-for-byte by the expiry regression). Recompute both from
    the retained cache with the current working candidate on each quality update, mirroring coverage, so that
    `max_sample_lifespan_us` strictly bounds fitness history. Optimizer history in the online parameters and the
    gravity projection dilutes through the floored learning rate and stays non-strict by design.
  - **Position:** `src/fusion/mag_calibrator.rs (cache_derived_fitness_statistics)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_online_history_outlives_sample_lifespan,`
    rewritten, plus a new partial-expiry sequence-equivalence test)
