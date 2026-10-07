## Medium severity

- [x] replace_fusion_inconsistency_with_consistency
  - **Summary:** `FusionInconsistency::inconsistency()` reduces estimator health to the unnormalized, post-blend
      average of correction magnitudes (no per-source verdict, no noise normalization, no filtering); replace it with
      the ArduPilot/PX4-style `Consistency` report and delete the legacy counter plumbing.
  - **Position:** src/fusion/inconsistency.rs (replace_fusion_inconsistency_with_consistency)
  - **Unit test:** src/fusion/consistency_tests.rs (replace_fusion_inconsistency_with_consistency)
