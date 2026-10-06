use nalgebra::{UnitQuaternion, Vector3};

use super::naive_cf::NaiveCF;
use super::{Fusion, MagCalibrator};

fn frd_to_rub(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.y, -v.z, -v.x)
}

#[test]
fn update_mag_uses_shared_mag_calibrator() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    let calibrated_north = Vector3::new(1.0, 0.0, 0.0);
    let raw_north = offset + scale.component_mul(&calibrated_north);
    let north_rub = frd_to_rub(raw_north);

    fusion.integrate_mag(&north_rub, true, 0);

    assert!(fusion.state.consistency.sources.mag.innovation.last < 0.001);
    assert!(fusion.state.attitude.angle() < 0.001);
}

/// Asserts the attitude keeps a level down axis within `tolerance`. A yaw
/// (heading) component passes by construction — only roll/pitch tilts are
/// checked, which is exactly the error the dip partition must prevent the
/// magnetometer from ever producing.
#[track_caller]
fn assert_level(attitude: &UnitQuaternion<f32>, tolerance: f32) {
    let down_body = attitude.inverse() * Vector3::z();
    assert!(
        (down_body - Vector3::z()).norm() < tolerance,
        "attitude not level: {attitude:?}"
    );
}

#[test]
fn update_mag_estimates_magnetic_dip_angle() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 60 deg below the horizon, its horizontal component is true
    // north, and no gravity hint is passed, so the calibrator's dip
    // projection stays unseeded and the refinement runs cold from zero
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    // The dip state converges in $\sin\delta$ at (1 - BASE_DIP_RATIO) per
    // sample, and the heading step may only ever yaw towards the azimuthal
    // residual of the fitted calibration: the level must stay exact
    // throughout, where the old full-vector correction instead tipped the
    // attitude by the unmodeled 60 deg dip.
    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, false, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }

    // equilibrium offset is the calibration direction error projected onto
    // the meridian, bounded by the 0.01 rad resolution observed in
    // `update_mag_uses_shared_mag_calibrator`'s single-step innovation
    let dip_sin = fusion.state.mag_dip_sin.expect("dip refined per sample");
    assert!(
        (dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "mag_dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );
}

#[test]
fn update_mag_seeds_dip_from_calibrated_gravity_projection() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 60 deg below the horizon; the gravity hint of the first
    // integration seeds the calibrator's dip projection, so the dip state
    // is seeded at the first published result instead of converging cold
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    fusion.integrate_mag(&dipped_rub, true, 0);
    let dip_sin = fusion
        .state
        .mag_dip_sin
        .expect("dip seeded on first result");
    // The one-shot seed is bias-limited by the working fit's preconditioner
    // gap: $\kappa/(\gamma r) = g^T A_w^{-1} A m$ reaches the exact dip only
    // as $A_w \to A$, and this fixture's strong (3, 2, 1.5) anisotropy
    // leaves a measured 2.4 deg seed error at 60 deg dip; 3 deg bounds that
    // with margin while still requiring the seed to skip the cold-start
    // transient. The per-sample refinement below asserts the tight value.
    assert!(
        (dip_sin.asin() - dip).abs() < 3.0f32.to_radians(),
        "seeded mag_dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );

    // continued refinement must not wander off the seeded value either
    for t in 1..800 {
        fusion.integrate_mag(&dipped_rub, true, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }
    let dip_sin = fusion.state.mag_dip_sin.expect("dip refined per sample");
    assert!(
        (dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "refined mag_dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );
}

#[test]
fn update_mag_dip_update_is_robust_to_yaw_error() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 30 deg below the horizon; initial yaw is 180 deg off, so
    // the horizontal component of the body reading points at body -x: the
    // measured and estimated north agree and the attitude estimate is a
    // (wrong-heading) fixed point of the heading step. The vertical
    // component of the field is yaw-invariant, so the dip refinement —
    // run cold without gravity hints — must still converge with the
    // correct sign, on the very samples where a meridional-tangent
    // formulation would push the estimate the wrong way.
    fusion.state.attitude = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::PI);
    let dip = 30.0f32.to_radians();
    let world_dipped = Vector3::new(dip.cos(), 0.0, dip.sin());
    let body_dipped = fusion.state.attitude.inverse() * world_dipped;
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&body_dipped));

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, false, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }

    let dip_sin = fusion.state.mag_dip_sin.expect("dip refined per sample");
    assert!(
        (dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "mag_dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );
}

#[test]
fn update_mag_skips_heading_near_the_magnetic_poles() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.consistency.sources.mag = Default::default();

    // field ~0.6 deg off vertical: horizontal norm 0.01 is below
    // MIN_HEADING_FIELD_NORM, so its azimuth is gated as pure noise
    let near_vertical = Vector3::new(0.01, 0.0, (1.0_f32 - 1.0e-4).sqrt());
    fusion.state.attitude =
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2);
    let initial = fusion.state.attitude;
    let near_vertical_rub = frd_to_rub(offset + scale.component_mul(&near_vertical));

    for t in 0..800 {
        fusion.integrate_mag(&near_vertical_rub, true, t);
    }

    // no heading innovation was ever recorded and the yaw offset survives
    assert_eq!(fusion.state.consistency.sources.mag.innovation.last, 0.0);
    assert!(
        (fusion.state.attitude * initial.inverse()).angle() < 1.0e-6,
        "attitude moved from {initial:?} to {:?}",
        fusion.state.attitude
    );
    // the dip step still runs on the gated reading: the estimate must
    // converge to the near-polar dip of the vertical field
    let dip_sin = fusion.state.mag_dip_sin.expect("dip refined per sample");
    assert!(
        dip_sin > 85.0f32.to_radians().sin(),
        "mag_dip_sin={}",
        dip_sin
    );
}

#[test]
fn update_mag_discards_ill_conditioned_calibration() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    *fusion.state.mag_calibrator = nearly_collinear_calibrator();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    let raw_mag = Vector3::new(10.0005, -4.9990, 3.00025);
    let mag_rub = frd_to_rub(raw_mag);

    fusion.integrate_mag(&mag_rub, true, 0);

    assert_eq!(fusion.state.consistency.sources.mag.innovation.last, 0.0);
    assert_eq!(fusion.state.consistency.sources.mag.innovation.ema, 0.0);
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
    let azimuth = 0.37 + i as f32 * 1.21;
    let z = -0.8 + 1.6 * i as f32 / 1022.0;
    let xy_radius = (1.0 - z * z).sqrt();
    Vector3::new(xy_radius * azimuth.cos(), xy_radius * azimuth.sin(), z)
}
