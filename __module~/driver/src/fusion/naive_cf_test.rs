use nalgebra::{UnitQuaternion, Vector3};

use super::mag_calibrator::MagCalibrator;
use super::naive_cf::NaiveCF;
use super::Fusion;

fn frd_to_rub(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.y, -v.z, -v.x)
}

#[test]
fn integrate_no_roll_skips_when_factor_is_none() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let attitude = UnitQuaternion::from_euler_angles(0.8, -0.4, 1.1);
    fusion.state.attitude = attitude;

    fusion.integrate_regress_roll();

    assert!(fusion.state.attitude.angle_to(&attitude) < 1.0e-5);
}

#[test]
fn update_mag_uses_shared_mag_calibrator() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    fusion.state.magCalibrator = Box::new(seeded_calibrator(offset, scale));
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.corrections.mag = Default::default();

    let calibrated_north = Vector3::new(1.0, 0.0, 0.0);
    let raw_north = offset + scale.component_mul(&calibrated_north);
    let north_rub = frd_to_rub(raw_north);

    fusion.integrate_mag(&north_rub, true, true, 0);

    assert!(fusion.state.corrections.mag.prev < 0.001);
    assert!(fusion.state.attitude.angle() < 0.001);
}

#[test]
fn update_mag_ignores_magnetic_dip_angle() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    fusion.state.magCalibrator = Box::new(seeded_calibrator(offset, scale));
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.corrections.mag = Default::default();

    // field dips 60 deg below the horizon, but its horizontal component is still true north
    let dipped_north = Vector3::new(0.5, 0.0, 0.75_f32.sqrt());
    let raw_north = offset + scale.component_mul(&dipped_north);
    fusion.integrate_mag(&frd_to_rub(raw_north), true, true, 0);

    assert!(fusion.state.corrections.mag.prev < 0.001);
    assert!(fusion.state.attitude.angle() < 0.001);
}

#[test]
fn update_mag_discards_ill_conditioned_calibration() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.magCalibrator = Box::new(nearly_collinear_calibrator());
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.corrections.mag = Default::default();

    let raw_mag = Vector3::new(10.0005, -4.9990, 3.00025);
    let mag_rub = frd_to_rub(raw_mag);

    fusion.integrate_mag(&mag_rub, true, true, 0);

    assert_eq!(fusion.state.corrections.mag.prev, 0.0);
    assert_eq!(fusion.state.corrections.mag.avg, 0.0);
    assert_eq!(fusion.state.attitude.angle(), 0.0);
}

fn seeded_calibrator(offset: Vector3<f32>, scale: Vector3<f32>) -> MagCalibrator<1023> {
    let mut calibrator = MagCalibrator::new();
    for i in 0..1023 {
        let direction = sample_direction(i);
        let _ =
            calibrator.evaluate_correct(offset + scale.component_mul(&direction), None, i as u64);
    }
    calibrator
}

fn nearly_collinear_calibrator() -> MagCalibrator<1023> {
    let mut calibrator = MagCalibrator::new();
    for i in 0..1023 {
        let t = i as f32 * 0.0001;
        let _ = calibrator.evaluate_correct(
            Vector3::new(10.0 + t, -5.0 + 2.0 * t, 3.0 + 0.5 * t),
            None,
            i as u64,
        );
    }
    calibrator
}

/// `NaiveCF` embeds the ~104 KB `MagCalibrator<1023>` inline in `FusionState`
/// and is constructed by value through `Default::default()` -> `new()` ->
/// `FusionState::new()` -> `NaiveCF::new()` -> `Box::new`. In debug builds
/// each layer keeps its own copy (plus one temporary per large array field)
/// live on the stack, peaking above 1 MiB; `examples/sensor_fusion.rs`
/// overflows the 1 MiB Windows main-thread stack inside `any_cf()`, before
/// the first `update()`. The struct must become pointer-sized so by-value
/// constructor moves stay cheap. This assertion fails cleanly while the
/// calibrator is stored inline.
#[test]
fn naive_cf_is_small_enough_for_by_value_construction() {
    assert!(
        std::mem::size_of::<NaiveCF>() <= 1024,
        "NaiveCF is {} bytes; by-value constructor moves overflow a 1 MiB \
         main-thread stack in debug builds",
        std::mem::size_of::<NaiveCF>()
    );
}

/// End-to-end reproduction of the `sensor_fusion.rs` overflow: construction
/// plus a mixed acc/gyro/mag update stream must fit in a bounded stack.
/// With the calibrator stored inline this aborts the test process with a
/// stack overflow (debug construction peaks above 1 MiB) rather than failing
/// cleanly. With it boxed, the measured debug peak is 256-320 KiB: one
/// ~104 KB calibrator instance plus its per-field temporaries during
/// `default()`, then the update path's ~64-128 KiB.
#[test]
fn construction_and_update_fit_bounded_stack() {
    const STACK_BUDGET: usize = 384 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK_BUDGET)
        .spawn(|| {
            let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
            for _ in 0..20 {
                fusion.update();
            }
        })
        .expect("spawn fusion thread")
        .join()
        .expect("fusion thread panicked");
}

fn sample_direction(i: usize) -> Vector3<f32> {
    let theta = 0.37 + i as f32 * 1.21;
    let z = -0.8 + 1.6 * i as f32 / 1022.0;
    let radius = (1.0 - z * z).sqrt();
    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}
