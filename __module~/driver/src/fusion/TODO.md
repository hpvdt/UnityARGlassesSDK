## Medium severity

- [ ]  Score replacement candidates in their post-replacement buffer

    - **Summary:** Candidate scoring excludes the replacement victim; the regression is enabled, pending merge.
    - **Position:** `src/fusion/mag/mag_calibrator.rs (candidate_score_includes_replaced_victim)`
    - **Unit test:** `src/fusion/mag/mag_calibrator_test.rs (mag_calibrator_candidate_score_includes_replaced_victim)`
