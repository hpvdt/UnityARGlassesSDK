//!
//! ArduPilot/PX4-style estimator consistency reporting for the fusion pipeline (draft).
//!
//! Both flight stacks report estimator health through *innovation consistency checks*: for each
//! observation source ("aiding source" in PX4) the innovation $\nu$ (residual between
//! measurement and prediction) is squared against its expected variance $\Sigma$ and an
//! innovation gate $\eta$ in multiples of the innovation standard deviation $\sqrt{\Sigma}$:
//!
//! $$\rho = \frac{\nu^2}{\eta^2 \, \Sigma}$$
//!
//! A sample with `test_ratio` $\rho \ge 1$ fails the check (ArduPilot `magTestRatio`,
//! `yawTestRatio`, `velTestRatio`, `posTestRatio`, `hgtTestRatio`; PX4 aid-source `test_ratio`),
//! and the source is healthy while the ratio stays below 1 (ArduPilot `magHealth`,
//! `velCheckPassed`, `posCheckPassed`). The filtered value $\bar{\rho}$ is the published
//! estimator-health signal: PX4's aid-source `test_ratio_filtered`, which `estimator_status`
//! exports as `*_test_ratio = sqrt(max(|test_ratio_filtered|))`, and ArduPilot's
//! `EKF_STATUS_REPORT` `*_variance` fields, exported as $\sqrt{\rho}$.
//!
//! [`SourceConsistency`] is estimator-agnostic:
//!
//! - complementary filter (e.g. `NaiveCF`): $\nu$ is the pre-correction angular residual and
//!   $\Sigma$ a configured innovation variance. [`SourceConsistency::record_scaled`]
//!   reconstructs $\nu$ from the post-blend correction the filter already stores in
//!   `NineAxis<Correction>`: `UnitQuaternion::scaled_rotation_between` scales the rotation
//!   angle exactly, so the residual is the applied correction divided by the blend ratio (code
//!   constants `BASE_GRAV_RATIO`, `BASE_MAG_RATIO`).
//! - EKF/ESKF: $\nu$ is the observation innovation and $\Sigma$ the live innovation variance;
//!   [`SourceConsistency::record_with_variance`] consumes both per sample.
//!
//! [`Consistency`] aggregates the acc/gyro/mag sources and fuses their verdicts, mirroring the
//! per-source fields of PX4 `EstimatorAidSource1d.msg` (`observation`/`observation_variance` are
//! deliberately omitted: the measurement and prediction live in the estimator, this tracker
//! only sees innovations) and ArduPilot's `NavEKF3_core` `*TestRatio`/`*InnovGate` members.
//! Angular magnitudes are frame-free scalars; the residuals derive from FRD vectors per the
//! module convention. The report is not yet wired into `Fusion`: when `NaiveCF` and a future
//! EKF both fill one, a `Fusion::consistency()` accessor supersedes `Fusion::corrections()`.
//!
use std::fmt;

use super::NineAxis;

#[cfg(test)]
#[path = "consistency_tests.rs"]
mod consistency_tests;

/// One scalar quantity with exponential moving average tracking: [`EmaTracking::last`] is the
/// most recent sample, [`EmaTracking::ema`] its exponential moving average. Both are updated
/// together by [`EmaTracking::record`]; updating only one by hand breaks the pairing.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmaTracking {
    /// Most recently recorded value.
    pub last: f32,
    /// Exponential moving average of all recorded values.
    pub ema: f32,
}

impl EmaTracking {
    /// Exponential averaging decay per sample, shared with the legacy `Correction` counters.
    /// At factor 0.9 the filter's continuous-equivalent time constant is
    /// $\tau \approx -dt / \ln 0.9 \approx 9.5$ samples ($\approx 0.48$ s at a 20 Hz
    /// magnetometer rate, $\approx 0.1$ s at 100 Hz). PX4's `test_ratio_filtered` instead uses
    /// a dt-dependent gain $\alpha = dt / (dt + \tau)$ with a fixed $\tau = 0.5$ s, so its
    /// memory is rate-independent; switching this to a dt-dependent gain is a refinement for
    /// when sample timestamps are wired in.
    pub const AVG_DECAY: f32 = 0.90;

    /// Create a tracker seeded with `initial` as both latest value and average: used for
    /// configured quantities (e.g. a complementary filter's innovation variance) that are
    /// already known before the first sample.
    pub fn new(initial: f32) -> Self {
        Self {
            last: initial,
            ema: initial,
        }
    }

    /// Record one sample: `last` takes it and `ema` folds it in by [`Self::AVG_DECAY`].
    pub fn record(&mut self, value: f32) {
        self.last = value;
        self.ema = self.ema * Self::AVG_DECAY + value * (1.0 - Self::AVG_DECAY);
    }
}

/// Consistency verdict of one observation source or of the whole estimator. Mirrors the
/// pass/fail semantics of ArduPilot's `magHealth`/`velCheckPassed` health booleans and PX4's
/// `estimator_status.innovation_check_flags` bitmask; `Pending` is the startup state both
/// stacks implement by waiting for a few samples (ArduPilot `_mag_counter > 3`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConsistencyStatus {
    /// Fewer than [`SourceConsistency::MIN_SAMPLES`] innovations recorded; no verdict yet.
    #[default]
    Pending,
    /// Filtered test ratio $\bar{\rho} < 1$.
    Consistent,
    /// Filtered test ratio $\bar{\rho} \ge 1$: the source disagrees with the estimator by more
    /// than the gated innovation spread $\eta \sqrt{\Sigma}$.
    Inconsistent,
}

/// Innovation consistency statistics of one observation source ("aiding source" in PX4 terms):
/// the per-source row reported by ArduPilot `EKF_STATUS_REPORT` and PX4
/// `EstimatorAidSource1d/2d/3d.msg` + `estimator_status`. Field groups match their
/// ArduPilot/PX4 counterparts (`innovation.last` is PX4 `innovation`, `innovation.ema` is PX4
/// `innovation_filtered`, and so on). Angular sources (all three in the 9-axis pipeline) use
/// radians.
#[derive(Clone, Copy, Debug)]
pub struct SourceConsistency {
    /// Innovation magnitude $\nu$ (pre-correction residual), in source units: `last` is the
    /// latest sample, `ema` its filtered value.
    ///
    /// Counterparts: ArduPilot `innovMag`/`innovYaw`/`innovVelPos`/`innovVtas`, PX4
    /// `innovation`/`innovation_filtered` (theirs a first-order filter with dt-dependent gain,
    /// here a fixed exponential average; ArduPilot has no filtered innovation).
    pub innovation: EmaTracking,

    /// Test ratio $\rho = \nu^2 / (\eta^2 \Sigma)$ (unitless, already gate-normalized): `last`
    /// is the latest sample, `ema` the filtered value; the consistency verdict compares `ema`
    /// against 1.0.
    ///
    /// Counterparts: `last` — ArduPilot `magTestRatio`/`yawTestRatio`/`velTestRatio`/
    /// `posTestRatio`/`hgtTestRatio` = `sq(innov) / (sq(gate) * varInnov)`, PX4 `test_ratio` =
    /// `sq(innovation) / (sq(innovation_gate) * innovation_variance)`; `ema` — PX4
    /// `test_ratio_filtered` (signed there, unsigned here) and the `estimator_status`
    /// `*_test_ratio` exports `sqrt(max(|test_ratio_filtered|))`; ArduPilot exports
    /// `sqrt(*TestRatio)` as the `EKF_STATUS_REPORT` `*_variance` fields.
    pub test_ratio: EmaTracking,

    /// Innovation variance $\Sigma$ (the expected squared spread of the innovation), in
    /// squared source units. For a complementary filter this is configured once via
    /// [`Self::new`] ($\Sigma = \sigma^2$ of the configured source-unit noise $\sigma$) and
    /// `last`/`ema` both hold it; for a Kalman filter [`Self::record_with_variance`] records
    /// the live variance per sample, `last` keeping the latest and `ema` the tracked spread.
    ///
    /// Counterparts: ArduPilot `varInnovMag`/`varInnovVelPos`/`varInnov`, PX4
    /// `innovation_variance`.
    pub innovation_variance: EmaTracking,

    /// Innovation consistency gate $\eta$ in multiples of $\sqrt{\Sigma}$, floored at
    /// [`Self::MIN_INNOVATION_GATE`]. Configured per source; not sample-tracked.
    ///
    /// Counterparts: ArduPilot `EK3_*_I_GATE` parameters (`_magInnovGate`, `_yawInnovGate`,
    /// `_gpsVelInnovGate`, `_gpsPosInnovGate`, in centi-standard-deviations, floored at 1 via
    /// `MAX(0.01f * _magInnovGate, 1.0f)`), PX4 `innovation_gate` (parameter `EKF2_MAG_GATE`,
    /// floored at 1 via `math::max(_params.ekf2_mag_gate, 1.f)`).
    pub innovation_gate: f32,

    /// Whether the latest sample failed the consistency check ($\rho \ge 1$); a Kalman filter
    /// would refuse to fuse it (ArduPilot `fuse*Data = false`, `posCheckPassed`).
    ///
    /// Counterpart: PX4 `innovation_rejected`.
    pub innovation_rejected: bool,

    /// Total innovations recorded since construction.
    pub samples_count: u64,

    /// Innovations that failed the consistency check since construction (ArduPilot ages them
    /// via `last*PassTime_ms` timeout bookkeeping instead of counting).
    pub rejected_count: u64,
}

impl SourceConsistency {
    /// Minimum recorded samples before [`Self::status`] leaves [`ConsistencyStatus::Pending`].
    /// Mirrors ArduPilot's `_mag_counter > 3` startup guard.
    pub const MIN_SAMPLES: u64 = 5;

    /// Default innovation gate: 3 standard deviations, the ArduPilot `EK3_MAG_I_GATE` /
    /// `EK3_YAW_I_GATE` default (300 centi-$\sigma$).
    pub const DEFAULT_INNOVATION_GATE: f32 = 3.0;

    /// Lower bound of the innovation gate: both stacks floor it at one standard deviation
    /// (ArduPilot `MAX(0.01f * _magInnovGate, 1.0f)`, PX4 `math::max(ekf2_mag_gate, 1.f)`).
    pub const MIN_INNOVATION_GATE: f32 = 1.0;

    /// Numerical floor of `innovation_variance`, guarding the test-ratio division against a
    /// zero or negative configured spread; both stacks floor the innovation variance at the
    /// measurement-noise variance (`if (varInnovMag < R_MAG)`) rather than at zero, so
    /// production configurations stay orders of magnitude above this guard.
    pub const MIN_INNOVATION_VARIANCE: f32 = 1e-12;

    /// Generic starting innovation variance (rad$^2$, i.e. $\sigma = 0.2$ rad) so `Default`
    /// construction stays usable; production pipelines should prefer per-source variances,
    /// e.g. [`Consistency::attitude_defaults`].
    pub const DEFAULT_INNOVATION_VARIANCE: f32 = 0.04;

    /// Create a tracker for a source with the given innovation variance $\Sigma$ in squared
    /// source units.
    pub fn new(innovation_variance: f32) -> Self {
        Self {
            innovation: EmaTracking::default(),
            test_ratio: EmaTracking::default(),
            innovation_variance: EmaTracking::new(innovation_variance),
            innovation_gate: Self::DEFAULT_INNOVATION_GATE,
            innovation_rejected: false,
            samples_count: 0,
            rejected_count: 0,
        }
    }

    /// Configure the innovation gate $\eta$ in multiples of $\sqrt{\Sigma}$ (ArduPilot
    /// `EK3_*_I_GATE`, PX4 `EKF2_*_GATE`).
    pub fn innovation_gate(mut self, innovation_gate: f32) -> Self {
        self.innovation_gate = innovation_gate;
        self
    }

    /// Record an innovation $\nu$ against the configured variance and gate, and return the
    /// updated verdict. The returned status lets a complementary filter adapt its blend ratio
    /// online without a second read.
    pub fn record(&mut self, innovation: f32) -> ConsistencyStatus {
        self.record_gated(innovation, self.innovation_variance.ema)
    }

    /// Record an innovation reconstructed from a complementary filter's post-blend correction:
    /// `scaled` is the applied correction magnitude and `blend_ratio` the filter's correction
    /// weight ($1 - \mathrm{ratio}$, code constants `BASE_GRAV_RATIO`/`BASE_MAG_RATIO`), so the
    /// pre-correction residual is $\nu = $ `scaled / blend_ratio`.
    ///
    /// Returns `None` without recording when `blend_ratio` is not positive: the filter is then
    /// ignoring the source entirely and there is no innovation to report.
    pub fn record_scaled(&mut self, scaled: f32, blend_ratio: f32) -> Option<ConsistencyStatus> {
        if blend_ratio <= 0.0 {
            return None;
        }
        Some(self.record(scaled / blend_ratio))
    }

    /// Record an innovation with its live innovation variance $\Sigma$ (Kalman filter path):
    /// the sample is gated by the live variance rather than by the tracked one, and the stored
    /// `innovation_variance` tracks the live spread so reporting stays meaningful while the
    /// filter covariance breathes.
    ///
    /// Returns `None` without recording for a non-finite or non-positive variance, which
    /// carries no meaningful normalization.
    pub fn record_with_variance(
        &mut self,
        innovation: f32,
        innovation_variance: f32,
    ) -> Option<ConsistencyStatus> {
        if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
            return None;
        }
        self.innovation_variance.record(innovation_variance);
        Some(self.record_gated(innovation, self.innovation_variance.last))
    }

    /// Filtered verdict: [`ConsistencyStatus::Pending`] before [`Self::MIN_SAMPLES`], then
    /// [`ConsistencyStatus::Inconsistent`] while $\bar{\rho} \ge 1$ and
    /// [`ConsistencyStatus::Consistent`] below, the same threshold both stacks use for their
    /// per-source health booleans (`magHealth = magTestRatio < 1.0f && ...`,
    /// `velTestRatio < 1.0f`). The exponential average already damps single-sample flapping
    /// across the boundary; a stronger hysteresis can be layered on by the caller if devices
    /// near the threshold prove unstable.
    pub fn status(&self) -> ConsistencyStatus {
        if self.samples_count < Self::MIN_SAMPLES {
            return ConsistencyStatus::Pending;
        }
        if self.test_ratio.ema >= 1.0 {
            ConsistencyStatus::Inconsistent
        } else {
            ConsistencyStatus::Consistent
        }
    }

    fn record_gated(&mut self, innovation: f32, innovation_variance: f32) -> ConsistencyStatus {
        let innovation_variance = innovation_variance.max(Self::MIN_INNOVATION_VARIANCE);
        let innovation_gate = self.innovation_gate.max(Self::MIN_INNOVATION_GATE);
        self.samples_count += 1;
        self.innovation.record(innovation);
        // sq(innovation) / (sq(gate) * variance), as in both stacks
        self.test_ratio
            .record((innovation / innovation_gate).powi(2) / innovation_variance);
        self.innovation_rejected = self.test_ratio.last > 1.0;
        if self.innovation_rejected {
            self.rejected_count += 1;
        }
        self.status()
    }
}

impl Default for SourceConsistency {
    fn default() -> Self {
        Self::new(Self::DEFAULT_INNOVATION_VARIANCE)
    }
}

impl fmt::Display for SourceConsistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "innovation={:8.5}, innovation_filtered={:8.5}, test_ratio={:7.4}, \
             test_ratio_filtered={:7.4}, rejected={}/{}",
            self.innovation.last,
            self.innovation.ema,
            self.test_ratio.last,
            self.test_ratio.ema,
            self.rejected_count,
            self.samples_count
        )
    }
}

/// Fusion-wide consistency report: one [`SourceConsistency`] per 9-axis source plus the fused
/// verdicts, matching the shape of the legacy `NineAxis<Correction>` counters it supersedes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Consistency {
    /// Per-source innovation statistics, mapped by attitude role:
    /// - `acc`: gravity alignment innovation of the accelerometer (roll/pitch);
    /// - `gyro`: dead-reckoning increment magnitude of the gyroscope, kept as a sensor-health
    ///   signal — a complementary filter has no gyro correction, and an ESKF would instead
    ///   expose the quality of its delta-angle/bias states here;
    /// - `mag`: heading innovation of the calibrated magnetometer (yaw).
    pub sources: NineAxis<SourceConsistency>,
}

impl Consistency {
    /// Starting innovation variances for the 9-axis attitude pipeline: $\sigma = 0.2$ rad
    /// accelerometer gravity alignment, $\sigma = 0.15$ rad per-sample gyroscope increment
    /// (sensor-clipping domain, not motion noise), $\sigma = 0.5$ rad magnetic heading (loose
    /// until calibration publishes). Expect per-device tuning, mirrored from observed hardware
    /// noise.
    pub fn attitude_defaults() -> Self {
        Self {
            sources: NineAxis {
                acc: SourceConsistency::new(0.2 * 0.2),
                gyro: SourceConsistency::new(0.15 * 0.15),
                mag: SourceConsistency::new(0.5 * 0.5),
            },
        }
    }

    /// Worst verdict across the three sources: any [`ConsistencyStatus::Inconsistent`] sinks
    /// the estimate; otherwise a still-[`ConsistencyStatus::Pending`] source blocks the clean
    /// bill of health.
    pub fn status(&self) -> ConsistencyStatus {
        let statuses = [
            self.sources.acc.status(),
            self.sources.gyro.status(),
            self.sources.mag.status(),
        ];
        if statuses.contains(&ConsistencyStatus::Inconsistent) {
            ConsistencyStatus::Inconsistent
        } else if statuses.contains(&ConsistencyStatus::Pending) {
            ConsistencyStatus::Pending
        } else {
            ConsistencyStatus::Consistent
        }
    }

    /// Largest filtered test ratio $\bar{\rho}$ across sources; the MAVLink-style overall
    /// consistency score. Both stacks export $\sqrt{\max \bar{\rho}}$ per source at the
    /// MAVLink boundary (ArduPilot `EKF_STATUS_REPORT`, PX4 `estimator_status`), leaving the
    /// stored fields squared; apply `sqrt` at export time to match them exactly.
    pub fn worst_test_ratio(&self) -> f32 {
        self.sources
            .acc
            .test_ratio
            .ema
            .max(self.sources.gyro.test_ratio.ema)
            .max(self.sources.mag.test_ratio.ema)
    }
}

impl fmt::Display for Consistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sources)
    }
}
