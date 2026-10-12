# AGENTS.md - Fusion Module Guide

Read [README.md](README.md) for estimator behavior, calibration derivations, and the known adaptation limitation.
Use the driver's mandatory [acronym registry](../../AGENTS.md#acronyms) and [symbol registry](../../AGENTS.md#symbols).

## Editing constraints

- Preserve the frame and unit contracts of `GlassesEvent` and `Fusion`; keep transformations explicit and centralized.
- Keep working and published calibration state separate; preserve finite-input checks and last-known-good correction.
- Keep live quality statistics tied to retained cache support and document changes to optimizer-history semantics.

## Calibration validation

When changing the calibrator, preserve deterministic coverage for cache expiry and readiness, neighbor-cache
invariants, invalid magnetometer and gravity inputs, minibatch clamping, repeatability, full-SPD and asymmetric
distortion, degenerate samples, stable repeated correction, and last-known-good fallback. Run the focused checks first
from the driver crate root:

```bash
cargo test --package ar-drivers --no-default-features --lib fusion::mag::mag_calibrator::mag_calibrator_test
cargo test --package ar-drivers --no-default-features --lib fusion::_3dof::naive_cf_test
cargo test --package ar-drivers --no-default-features --test fusion regression -- --nocapture
```

Then run the applicable broad Rust checks from the [driver guide](../../AGENTS.md#validation).
