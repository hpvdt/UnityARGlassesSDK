#![cfg(feature = "xreal")]

use std::path::Path;

use ar_drivers::fusion::{rub_to_frd, MagCalibrator};
use ar_drivers::xreal_air::XrealAirReplay;
use ar_drivers::{ARGlasses, GlassesEvent};

/// Highest gravity loss allowed for a single post-warm-up evaluation: the
/// loss-domain equivalent of the former 0.7 gravity-fitness floor at the
/// default weight $w_g = 0.01$ (fitness 0.7 corresponds to a relative RMS
/// dip residual of 0.175, i.e. a mean square of $0.030625$ and a gravity
/// loss of $0.5 \cdot 0.01 \cdot 0.030625 = 0.000153125$). A disabled or
/// absent gravity term reports zero.
const MAX_GRAVITY_LOSS: f32 = 0.000153125;

/// Check actual published magnetic directions, independently of the optimizer's
/// gravity-projection surrogate. The three ten-second windows share one dip
/// angle after thirty seconds of warm-up, without adjusting the decoder's
/// factory rotation to this recording. Motion and online calibration contribute
/// residual dip noise, bounded here at ten degrees root-mean-square and three
/// degrees of window-mean drift.
#[test]
fn air1_factory_alignment_preserves_corrected_magnetic_dip() {
    let trace = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/xreal_air_air1_60s.log");
    let mut replay = XrealAirReplay::open(&trace).unwrap();
    let mut calibrator = MagCalibrator::<1023>::new();
    let mut gravity = None;
    let mut previous_timestamp = None;
    let mut first_timestamp = None;
    let mut magnetic_samples = 0;
    let mut angles: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut maximum_gravity_loss = 0.0f32;
    let mut gravity_loss_sum = 0.0f64;

    loop {
        let event = replay.read_event().unwrap();
        let timestamp = match event {
            GlassesEvent::AccGyro { timestamp, .. }
            | GlassesEvent::Magnetometer { timestamp, .. } => timestamp,
            _ => continue,
        };
        if previous_timestamp.is_some_and(|previous| timestamp < previous) {
            break;
        }
        previous_timestamp = Some(timestamp);
        let elapsed = timestamp - *first_timestamp.get_or_insert(timestamp);
        match event {
            GlassesEvent::AccGyro { accelerometer, .. } => {
                gravity = rub_to_frd(&accelerometer).try_normalize(0.0);
            }
            GlassesEvent::Magnetometer { magnetometer, .. } => {
                magnetic_samples += 1;
                let Some(gravity) = gravity else { continue };
                let result = calibrator
                    .evaluate_correct(rub_to_frd(&magnetometer), Some(gravity), timestamp)
                    .unwrap();
                if elapsed < 30_000_000 {
                    continue;
                }
                let direction = result
                    .direction
                    .expect("calibration was not ready after warm-up");
                let angle = direction.dot(&gravity).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(angle.is_finite(), "non-finite corrected dip at {timestamp}");
                let window = ((elapsed - 30_000_000) / 10_000_000) as usize;
                angles[window].push(f64::from(angle));
                assert!(
                    result.gravity_loss.is_finite(),
                    "non-finite gravity loss at {timestamp}"
                );
                maximum_gravity_loss = maximum_gravity_loss.max(result.gravity_loss);
                gravity_loss_sum += f64::from(result.gravity_loss);
            }
            _ => {}
        }
    }

    assert_eq!(
        magnetic_samples, 9253,
        "fresh magnetic observations were lost or duplicated"
    );
    let count: usize = angles.iter().map(Vec::len).sum();
    assert!(
        count > 4000,
        "not enough corrected observations after warm-up"
    );
    let mean_angle = angles.iter().flatten().sum::<f64>() / count as f64;
    let rms_deviation = (angles
        .iter()
        .flatten()
        .map(|angle| (angle - mean_angle).powi(2))
        .sum::<f64>()
        / count as f64)
        .sqrt();
    eprintln!(
        "Air 1 corrected dip: mean={mean_angle:.3} deg, RMS deviation={rms_deviation:.3} deg; \
         gravity loss mean={:.6}, maximum={maximum_gravity_loss:.6}",
        gravity_loss_sum / count as f64,
    );
    assert!(
        maximum_gravity_loss < MAX_GRAVITY_LOSS,
        "gravity loss rose to {maximum_gravity_loss}"
    );
    assert!(
        rms_deviation < 10.0,
        "corrected dip varied by {rms_deviation:.3} degrees RMS"
    );
    for (window, angles) in angles.iter().enumerate() {
        assert!(
            angles.len() > 1000,
            "window {window} has too few observations"
        );
        let window_mean = angles.iter().sum::<f64>() / angles.len() as f64;
        assert!(
            (window_mean - mean_angle).abs() < 3.0,
            "window {window} mean dip {window_mean:.3} differs from {mean_angle:.3} degrees",
        );
    }
}
