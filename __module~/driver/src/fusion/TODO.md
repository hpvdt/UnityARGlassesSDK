## Medium severity

- [ ]  Score replacement candidates in their post-replacement buffer

  - **Summary:** Candidate and victim diversity scores currently use different neighbor pools.
  - **Position:** `src/fusion/mag_calibrator.rs (candidate_score_includes_replaced_victim)`
  - **Unit test:** `src/fusion/mag_calibrator_test.rs (mag_calibrator_candidate_score_includes_replaced_victim)`
