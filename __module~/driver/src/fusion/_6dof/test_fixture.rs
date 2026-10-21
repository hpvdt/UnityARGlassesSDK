/*!
Shared test fixture of the 6-DoF estimator suites: the lockstep SimMotion
scoring harness and the frame-conversion helpers its suites share.

`run_sim_motion_scored` pairs an estimator's [`Fusion`] update loop against
a reference `SimMotion` fixture with an identical seed and configuration:
the reference advances its mirror state once per consumed event, so its
snapshot before each read matches the event the fusion last processed. An
unpaced 20.001 ms event period keeps the run fast in wall time while the
virtual timeline advances at ~50 Hz (AccGyro and Magnetometer alternating
at ~25 Hz each), and the run reports velocity and attitude metrics so
estimators sharing the `Fusion` interface benchmark against each other on
identical data.
*/

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use crate::fusion::{rub_to_frd, Consistency, Fusion};
use crate::sim::sim_motion;
use crate::sim::SimMotion;
use crate::ARGlasses;

/// Aggregate metrics of one scored SimMotion run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScoredRun {
    /// Mean estimated speed over the scored span, in m/s.
    pub(crate) mean_estimate_norm: f32,
    /// Mean true speed over the scored span, in m/s.
    pub(crate) mean_truth_norm: f32,
    /// Mean absolute velocity error over the first scored half, in m/s.
    pub(crate) mean_error_first: f32,
    /// Mean absolute velocity error over the second scored half, in m/s.
    pub(crate) mean_error_second: f32,
    /// Mean attitude angle error over the scored span, in radians: the
    /// rotation between the estimated and true body-to-world attitude.
    pub(crate) mean_attitude_error: f32,
    /// Consistency report at the end of the run.
    pub(crate) consistency: Consistency,
}

/// Drives `make_fusion`'s estimator against a lockstep SimMotion reference
/// and scores velocity and attitude over the `scored_events` events
/// following `warmup_events`, split into halves for the no-accumulation
/// check. The truth pairing follows the lockstep snapshot semantics: each
/// scored estimate compares against the state that produced the event the
/// fusion last processed.
pub(crate) fn run_sim_motion_scored(
    config: sim_motion::Config,
    warmup_events: usize,
    scored_events: usize,
    make_fusion: impl FnOnce(Box<dyn ARGlasses>) -> Box<dyn Fusion>,
) -> ScoredRun {
    let config = sim_motion::Config {
        event_period_us: 20_001, // just above SimMotion's 20 ms pacing bound
        ..config
    };
    let mut fusion = make_fusion(Box::new(SimMotion::with_config(config.clone())));
    let mut reference = SimMotion::with_config(config);

    let mut estimate_norm_sum = 0.0f32;
    let mut truth_norm_sum = 0.0f32;
    let mut attitude_error_sum = 0.0f32;
    let mut error_sum = [0.0f32; 2];
    let mut scored = [0u32; 2];
    for event in 0..(warmup_events + scored_events) {
        // snapshot state produced the event the fusion consumed last
        let snapshot = reference.snapshot();
        fusion.update();
        let _ = reference.read_event().unwrap();
        if event < warmup_events {
            continue;
        }

        let truth = rub_to_frd(&snapshot.velocity);
        let attitude_error = fusion
            .attitude_quaternion()
            .angle_to(&attitude_rub_to_frd(&snapshot.attitude));
        let half = (event - warmup_events) / (scored_events / 2);
        estimate_norm_sum += fusion.linear_velocity().norm();
        truth_norm_sum += truth.norm();
        attitude_error_sum += attitude_error;
        error_sum[half.min(1)] += (fusion.linear_velocity() - truth).norm();
        scored[half.min(1)] += 1;
    }

    let scored_total = (scored[0] + scored[1]) as f32;
    ScoredRun {
        mean_estimate_norm: estimate_norm_sum / scored_total,
        mean_truth_norm: truth_norm_sum / scored_total,
        mean_error_first: error_sum[0] / scored[0] as f32,
        mean_error_second: error_sum[1] / scored[1] as f32,
        mean_attitude_error: attitude_error_sum / scored_total,
        consistency: fusion.consistency(),
    }
}

/// Conjugates a body-RUB to world-RUB attitude into the fusion state's
/// body-FRD to world-FRD convention: the component orders of both the body
/// and world frames change by the same fixed rotation $c$, so the
/// conversion is the conjugation
/// $S_{\mathrm{FRD}} = c\,S_{\mathrm{RUB}}\,c^{-1}$. The frame-change
/// rotation $c$ is 120 deg about $(-1,-1,1)/\sqrt{3}$, whose quaternion
/// $(0.5, -0.5, -0.5, 0.5)$ has every component exactly representable in
/// f32.
pub(crate) fn attitude_rub_to_frd(attitude: &UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let c = UnitQuaternion::from_quaternion(Quaternion::new(0.5, -0.5, -0.5, 0.5));
    c * attitude * c.inverse()
}

/// Converts an FRD vector into the device's RUB frame, the inverse of
/// `rub_to_frd`, for crafting synthetic raw events.
pub(crate) fn frd_to_rub(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.y, -v.z, -v.x)
}

/// Felt gravity of a level device, the exact FRD reading at rest.
pub(crate) fn felt_gravity_frd() -> Vector3<f32> {
    Vector3::new(0.0, 0.0, -9.81)
}
