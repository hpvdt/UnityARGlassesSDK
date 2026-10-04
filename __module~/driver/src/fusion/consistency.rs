//!
//! ArduPilot/PX4-style estimator consistency reporting for the fusion pipeline (draft).
//!
//! Both flight stacks report estimator health through normalized innovations. For each
//! observation source, the residual magnitude $\nu$ between the measurement and the prediction
//! is divided by its expected spread $\sigma$, and the test ratio $\rho = (\nu / \sigma)^2$
//! plus its filtered value $\bar{\rho}$ are published (ArduPilot `EKF_STATUS_REPORT` variance
//! fields, PX4 `estimator_status` `*_test_ratio` fields). A filtered ratio below 1.0 means the
//! source agrees with the estimator within its declared noise level; sustained values at or
//! above 1.0 report inconsistency.
//!
//! [`SourceConsistency`] is estimator-agnostic:
//!
//! - complementary filter (e.g. `NaiveCF`): $\nu$ is the pre-correction angular residual and
//!   $\sigma$ a configured noise gate. [`SourceConsistency::record_scaled`] reconstructs $\nu$
//!   from the post-blend correction the filter already stores in `NineAxis<Correction>`:
//!   `UnitQuaternion::scaled_rotation_between` scales the rotation angle exactly, so the
//!   residual is the applied correction divided by the blend ratio (code constants
//!   `BASE_GRAV_RATIO`, `BASE_MAG_RATIO`).
//! - EKF/ESKF: $\nu$ is the observation innovation and $\sigma = \sqrt{\Sigma}$ the live
//!   innovation spread; [`SourceConsistency::record_with_variance`] consumes both per sample.
//!
//! [`Consistency`] aggregates the acc/gyro/mag sources and fuses their verdicts. Angular
//! magnitudes are frame-free scalars; the residuals derive from FRD vectors per the module
//! convention. The report is not yet wired into `Fusion`: when `NaiveCF` and a future EKF both
//! fill one, a `Fusion::consistency()` accessor supersedes `Fusion::corrections()`.
//!
use std::fmt;

use super::NineAxis;

#[cfg(test)]
#[path = "consistency_tests.rs"]
mod consistency_tests;

/// Consistency verdict of one observation source or of the whole estimator, mirroring the
/// pass/fail semantics of ArduPilot `EKF_STATUS_REPORT` and PX4 `estimator_status`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConsistencyStatus {
    /// Fewer than [`SourceConsistency::MIN_SAMPLES`] innovations recorded; no verdict yet.
    #[default]
    Pending,
    /// Filtered test ratio $\bar{\rho} < 1$.
    Consistent,
    /// Filtered test ratio $\bar{\rho} \ge 1$: the source disagrees with the estimator by more
    /// than its declared noise level.
    Inconsistent,
}

/// Innovation statistics of one observation source: the per-source row of an ArduPilot
/// `EKF_STATUS_REPORT`. Angular sources (all three in the 9-axis pipeline) use radians.
#[derive(Clone, Copy, Debug)]
pub struct SourceConsistency {
    /// Innovation gate $\sigma$: the expected 1-standard-deviation spread of the innovation
    /// in source units. Configured for a complementary filter; for a Kalman filter this field
    /// tracks the live innovation spread recorded through
    /// [`SourceConsistency::record_with_variance`].
    pub gate: f32,
    /// Latest innovation magnitude $\nu$ (pre-correction residual), in source units.
    pub latest: f32,
    /// Exponential moving average $\bar{\nu}$ of the innovation magnitude.
    pub avg: f32,
    /// Latest test ratio $\rho = (\nu / \sigma)^2$ (unitless, PX4 `*_test_ratio` semantics).
    pub test_ratio: f32,
    /// Exponential moving average $\bar{\rho}$ of the test ratio, the ArduPilot
    /// `EKF_STATUS_REPORT` variance fields' semantics; the filtered verdict threshold is 1.0.
    pub avg_test_ratio: f32,
    /// Total innovations recorded since construction.
    pub samples: u64,
    /// Innovations whose normalized magnitude exceeded [`SourceConsistency::REJECT_GATE`];
    /// a Kalman filter would have refused to fuse them.
    pub rejected: u64,
}

impl SourceConsistency {
    /// Exponential averaging decay per sample, shared with the legacy `Correction` counters.
    pub const AVG_DECAY: f32 = 0.90;

    /// Minimum recorded samples before [`Self::status`] leaves [`ConsistencyStatus::Pending`].
    pub const MIN_SAMPLES: u64 = 5;

    /// Normalized-innovation outlier gate: an innovation beyond 3 standard deviations occurs
    /// at ~0.3% rate under consistent Gaussian noise and counts as a rejected observation.
    pub const REJECT_GATE: f32 = 3.0;

    /// Numerical floor of `gate`, guarding the test-ratio division against a zero or
    /// negative configured spread. Far below any physical noise level; configure real gates
    /// orders of magnitude above it.
    pub const MIN_GATE: f32 = 1e-6;

    /// Generic starting gate (0.2 rad) so `Default` construction stays usable; production
    /// pipelines should prefer per-source gates, e.g. [`Consistency::attitude_defaults`].
    pub const DEFAULT_GATE: f32 = 0.2;

    /// Create a tracker with the given innovation gate $\sigma$ in source units.
    pub fn new(gate: f32) -> Self {
        Self {
            gate,
            latest: 0.0,
            avg: 0.0,
            test_ratio: 0.0,
            avg_test_ratio: 0.0,
            samples: 0,
            rejected: 0,
        }
    }

    /// Record an innovation $\nu$ against the configured gate and return the updated verdict.
    ///
    /// The returned status lets a complementary filter adapt its blend ratio online without a
    /// second read.
    pub fn record(&mut self, innovation: f32) -> ConsistencyStatus {
        self.record_gated(innovation, self.gate)
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
    /// the sample is gated by $\sqrt{\Sigma}$ rather than by the configured gate, and the
    /// stored `gate` tracks the live spread through the same exponential average so reporting
    /// stays meaningful while the filter covariance breathes.
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
        let live_gate = innovation_variance.sqrt();
        self.gate = self.gate * Self::AVG_DECAY + live_gate * (1.0 - Self::AVG_DECAY);
        Some(self.record_gated(innovation, live_gate))
    }

    /// Latest innovation in gate units ($\nu / \sigma$); values above [`Self::REJECT_GATE`]
    /// are outliers a Kalman filter would refuse to fuse.
    pub fn normalized_latest(&self) -> f32 {
        self.latest / self.gate.max(Self::MIN_GATE)
    }

    /// Whether the latest innovation exceeded [`Self::REJECT_GATE`].
    pub fn is_rejected(&self) -> bool {
        self.normalized_latest() > Self::REJECT_GATE
    }

    /// Filtered verdict: [`ConsistencyStatus::Pending`] before [`Self::MIN_SAMPLES`], then
    /// [`ConsistencyStatus::Inconsistent`] while $\bar{\rho} \ge 1$ and
    /// [`ConsistencyStatus::Consistent`] below. The exponential average already damps
    /// single-sample flapping across the boundary; a stronger hysteresis can be layered on by
    /// the caller if devices near the threshold prove unstable.
    pub fn status(&self) -> ConsistencyStatus {
        if self.samples < Self::MIN_SAMPLES {
            return ConsistencyStatus::Pending;
        }
        if self.avg_test_ratio >= 1.0 {
            ConsistencyStatus::Inconsistent
        } else {
            ConsistencyStatus::Consistent
        }
    }

    fn record_gated(&mut self, innovation: f32, gate: f32) -> ConsistencyStatus {
        let gate = gate.max(Self::MIN_GATE);
        self.samples += 1;
        self.latest = innovation;
        self.avg = self.avg * Self::AVG_DECAY + innovation * (1.0 - Self::AVG_DECAY);
        let normalized = innovation / gate;
        self.test_ratio = normalized * normalized;
        self.avg_test_ratio =
            self.avg_test_ratio * Self::AVG_DECAY + self.test_ratio * (1.0 - Self::AVG_DECAY);
        if normalized > Self::REJECT_GATE {
            self.rejected += 1;
        }
        self.status()
    }
}

impl Default for SourceConsistency {
    fn default() -> Self {
        Self::new(Self::DEFAULT_GATE)
    }
}

impl fmt::Display for SourceConsistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "innov={:8.5}, innov_avg={:8.5}, test_ratio={:7.4}, avg_test_ratio={:7.4}, \
             rejected={}/{}",
            self.latest,
            self.avg,
            self.test_ratio,
            self.avg_test_ratio,
            self.rejected,
            self.samples
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
    /// Starting gates for the 9-axis attitude pipeline: 0.2 rad accelerometer gravity
    /// alignment, 0.15 rad per-sample gyroscope increment (sensor-clipping domain, not motion
    /// noise), 0.5 rad magnetic heading (loose until calibration publishes). Expect per-device
    /// tuning, mirrored from observed hardware noise.
    pub fn attitude_defaults() -> Self {
        Self {
            sources: NineAxis {
                acc: SourceConsistency::new(0.2),
                gyro: SourceConsistency::new(0.15),
                mag: SourceConsistency::new(0.5),
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
    /// consistency score.
    pub fn worst_test_ratio(&self) -> f32 {
        self.sources
            .acc
            .avg_test_ratio
            .max(self.sources.gyro.avg_test_ratio)
            .max(self.sources.mag.avg_test_ratio)
    }
}

impl fmt::Display for Consistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sources)
    }
}
