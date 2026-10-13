use nalgebra::{UnitQuaternion, Vector3};

use super::naive_cf::NaiveCF;
use super::Fusion;
use crate::fusion::MagCalibrator;

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
fn integrate_gyro_propagates_attitude() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_gyro = (Vector3::zeros(), 0);

    // 100 ms of elapsed sensor time scales the rad/s reading into the
    // (0.06, -0.04, 0.02) rad dead-reckoning increment
    let gyro_frd = Vector3::new(0.6, -0.4, 0.2);
    fusion.integrate_gyro(&frd_to_rub(gyro_frd), 100_000);

    let increment = UnitQuaternion::from_euler_angles(0.06, -0.04, 0.02);
    assert!(fusion.state.attitude.angle_to(&increment) < 1.0e-6);
    assert_eq!(fusion.prev_gyro.1, 100_000);
    assert!((fusion.state.consistency.sources.gyro.innovation - increment.angle()).abs() < 1.0e-6);
}

#[test]
fn integrate_gyro_skips_nonpositive_dt() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let attitude = UnitQuaternion::from_euler_angles(0.8, -0.4, 1.1);
    fusion.state.attitude = attitude;
    fusion.prev_gyro = (Vector3::zeros(), 500_000);
    fusion.state.consistency.sources.gyro = Default::default();

    // duplicate timestamp: zero elapsed time must neither propagate nor record
    fusion.integrate_gyro(&frd_to_rub(Vector3::new(1.0, 1.0, 1.0)), 500_000);

    assert!(fusion.state.attitude.angle_to(&attitude) < 1.0e-7);
    assert_eq!(fusion.state.consistency.sources.gyro.innovation, 0.0);
    assert_eq!(fusion.prev_gyro.1, 500_000);
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

    fusion.integrate_mag(&north_rub, 0);

    assert!(fusion.state.consistency.sources.mag.innovation < 0.001);
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
fn update_mag_keeps_level_with_dipped_field() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 60 deg below the horizon and its horizontal component is
    // true north; the heading step may only ever yaw towards the azimuthal
    // residual of the fitted calibration, so the level must stay exact
    // throughout, where a full-vector correction would instead tip the
    // attitude by the unmodeled 60 deg dip
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }
}

#[test]
fn update_mag_gravity_hint_reports_dip() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 60 deg below the horizon with true-north horizontal
    // component, and the acc-style gravity hint (pointing up) is passed on
    // every integration: the calibrator's gravity surrogate learns the
    // projection $g^T m = -\sin\delta$ while the attitude level must stay
    // exact throughout
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }

    // the reported projection is bias-limited by the working fit's
    // preconditioner gap: $\kappa/(\gamma r) = g^T A_w^{-1} A m$ reaches
    // the exact dip only as $A_w \to A$, and this fixture's strong
    // (3, 2, 1.5) anisotropy leaves a measured 2.4 deg error at 60 deg
    // dip; 3 deg bounds that with margin
    let result = fusion
        .state
        .mag_calibrator
        .evaluate_correct(offset + scale.component_mul(&dipped_north), None, 800)
        .unwrap();
    let dip_sin = result.dip_sin.expect("dip reported with gravity hints");
    assert!(
        ((-dip_sin).asin() - dip).abs() < 3.0f32.to_radians(),
        "dip_sin={}, expected -sin({})",
        dip_sin,
        dip
    );

    // the fusion state's own estimate carries the same dip in the
    // opposite sign convention: positive below the horizon, against
    // the acc-style hint's up-pointing projection (tolerance basis:
    // the seeded fit's down-axis error at this dipped direction, as
    // measured in
    // `update_mag_initializes_dip_from_first_calibrated_reading`)
    let mag_dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!(
        (mag_dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "mag_dip_sin={}, expected sin({})",
        mag_dip_sin,
        dip
    );
}

#[test]
fn update_mag_initializes_dip_from_first_calibrated_reading() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // the seeded calibrator has never seen a gravity hint, so its
    // `dip_sin` record is absent before the first integration; the
    // fusion state's dip estimate must not depend on that record
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));
    let result = fusion
        .state
        .mag_calibrator
        .evaluate_correct(offset + scale.component_mul(&dipped_north), None, 0)
        .unwrap();
    assert_eq!(result.dip_sin, None);
    assert_eq!(fusion.state.mag_dip_sin, None);
    // (the probe also deposits one hint-less sample into the
    // already-full cache; the tolerance below absorbs the fit's
    // response)

    // the first usable reading initializes the dip state from the
    // field's vertical projection onto the estimated down axis; the
    // reading itself, not the calibrator's record, supplies the value
    fusion.integrate_mag(&dipped_rub, 0);

    // 1 deg basis: the seeded anisotropic fit leaves a measured ~0.85
    // deg asin gap at this 60 deg-dipped direction (its ~15 mrad
    // directional error carries a ~7.5 mrad down-axis component), so
    // the bound holds with ~1.2x margin on the measured fixture
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state initialized");
    assert!(
        (dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );
}

#[test]
fn update_mag_refines_dip_towards_true_dip() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // the dip state starts deliberately wrong (equator dip); the
    // per-sample complementary refinement must pull it towards the
    // 60 deg true dip the calibrated readings carry, converging by the
    // $\mathrm{ratio}^n$ factor of the blend (`BASE_DIP_RATIO` per step)
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));
    fusion.state.mag_dip_sin = Some(0.0);

    // one step moves the state by exactly the blend weight
    // $1 - \mathrm{ratio}$ (`BASE_DIP_RATIO`) of the residual,
    // 0.02 * sin(60 deg) ~ 0.017; the fit's ~7.5 mrad
    // down-axis error is scaled down by the same factor to ~1.4e-4
    // (measured 1.4e-4), so a full-replacement or zero-blend update
    // lands far outside the 1e-3 bound
    fusion.integrate_mag(&dipped_rub, 0);
    let first = fusion.state.mag_dip_sin.expect("dip state present");
    let ratio = crate::fusion::_3dof::naive_cf_core::BASE_DIP_RATIO;
    assert!(
        (first - ratio * dip.sin()).abs() < 1.0e-3,
        "first dip step {}, expected {} * sin(60 deg)",
        first,
        ratio
    );

    for t in 1..800 {
        fusion.integrate_mag(&dipped_rub, t);
    }

    // same 1 deg fit-error basis as the initialization test
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!(
        (dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "dip_sin={}, expected sin({})",
        dip_sin,
        dip
    );
}

#[test]
fn update_mag_dip_projects_onto_estimated_down_axis() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0);
    fusion.state.consistency.sources.mag = Default::default();

    // the true device is level with a 60 deg-dipped northern field, but
    // the estimate is 0.5 rad pitched: the reading (body frame = world
    // frame) projected onto the ESTIMATED down axis lands ~0.35 away
    // from the world-frame projection, because the pitched down axis
    // turns away from the dipping field. The heading step still runs
    // (the tilted level leaves a large horizontal component) but it
    // rotates about the estimated down axis, which it cannot move, so
    // the dip's fixed point stays the estimated-down projection alone
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));
    let expected = (fusion.state.attitude.inverse() * Vector3::z()).dot(&dipped_north);

    // guard the fixture's discriminating power: the two projections
    // must stay well separated
    assert!(
        (expected - dipped_north.z).abs() > 0.1,
        "fixture tilt too small to distinguish the projections: {expected}"
    );

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, t);
    }

    // the anisotropic fixture's fit leaves a measured ~0.006 converged
    // gap at this dipped direction (a mid-run transient ~0.012 while
    // the hinted samples flood the cache), so 0.02 bounds it with
    // margin while staying >17x tighter than the ~0.35 separation the
    // test discriminates on
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!(
        (dip_sin - expected).abs() < 2.0e-2,
        "dip_sin={dip_sin}, expected the estimated-down projection {expected}"
    );
}

/// Open issue: the attitude correction must compare the full calibrated
/// field against the dipped reference built from `mag_dip_sin`, instead
/// of the horizontal component alone; until then the level the acc
/// filter maintains is provably invariant under the mag correction.
#[test]
#[ignore = "open issue: the attitude correction ignores the learned dip state"]
fn update_mag_correction_tips_level_through_learned_dip() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.consistency.sources.mag = Default::default();

    // level true device with a 60 deg-dipped northern field, estimate
    // pitched 0.5 rad, dip state seeded to the true field: the
    // full-field correction must pull the level toward truth, which a
    // horizontal-only correction can never do (its rotation axis is
    // the estimated down axis, which leaves the tilt untouched)
    let dip = 60.0f32.to_radians();
    fusion.state.attitude = UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0);
    fusion.state.mag_dip_sin = Some(dip.sin());
    let tilt_before = ((fusion.state.attitude.inverse() * Vector3::z()) - Vector3::z()).norm();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    fusion.integrate_mag(&dipped_rub, 0);

    // one step moves the tilt by BASE_MAG_RATIO of the ~0.5 rad
    // reference/reading gap (>= 2e-3); the horizontal-only partition
    // leaves it exactly unchanged
    let tilt_after = ((fusion.state.attitude.inverse() * Vector3::z()) - Vector3::z()).norm();
    assert!(
        tilt_after < tilt_before - 1.0e-3,
        "level tilt unchanged at {tilt_after} (before {tilt_before}): the correction never used the dip"
    );
}

#[test]
fn update_mag_keeps_dip_across_unusable_readings() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));
    fusion.integrate_mag(&dipped_rub, 0);
    let initialized = fusion.state.mag_dip_sin.expect("dip state initialized");

    // a later unusable calibration (degenerate fit, no direction) must
    // not erase the learned dip state: the estimate outlives readings
    // that cannot feed it
    *fusion.state.mag_calibrator = nearly_collinear_calibrator();
    let raw_mag = Vector3::new(10.0005, -4.9990, 3.00025);
    fusion.integrate_mag(&frd_to_rub(raw_mag), 1);

    assert_eq!(fusion.state.mag_dip_sin, Some(initialized));
}

#[test]
fn update_mag_keeps_level_under_yaw_error() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 30 deg below the horizon; initial yaw is 180 deg off, so
    // the horizontal component of the body reading points at body -x: the
    // measured and estimated north agree and the attitude estimate sits at
    // a (wrong-heading) fixed point of the heading step — the very case
    // where a vertical or meridional-tangent correction would tip the
    // level
    fusion.state.attitude = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::PI);
    let dip = 30.0f32.to_radians();
    let world_dipped = Vector3::new(dip.cos(), 0.0, dip.sin());
    let body_dipped = fusion.state.attitude.inverse() * world_dipped;
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&body_dipped));

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, t);
        assert_level(&fusion.state.attitude, 1.0e-3);
    }
}

#[test]
fn update_mag_skips_heading_near_the_magnetic_poles() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.consistency.sources.mag = Default::default();

    // field ~0.6 deg off vertical: horizontal norm 0.01 is below
    // MIN_HEADING_MAG_FIELD_NORM, so its azimuth is gated as pure noise
    let near_vertical = Vector3::new(0.01, 0.0, (1.0_f32 - 1.0e-4).sqrt());
    fusion.state.attitude =
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2);
    let initial = fusion.state.attitude;
    let near_vertical_rub = frd_to_rub(offset + scale.component_mul(&near_vertical));

    for t in 0..800 {
        fusion.integrate_mag(&near_vertical_rub, t);
    }

    // no heading innovation was ever recorded and the yaw offset survives
    assert_eq!(fusion.state.consistency.sources.mag.innovation, 0.0);
    assert!(
        (fusion.state.attitude * initial.inverse()).angle() < 1.0e-6,
        "attitude moved from {initial:?} to {:?}",
        fusion.state.attitude
    );

    // the gated azimuth still carries full dip evidence: the dip state
    // refined from the near-vertical reading while the heading stayed
    // locked (1e-3 basis: the exact-fit calibration's directional
    // error is sub-mrad, leaving an order of magnitude of margin)
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!((dip_sin - near_vertical.z).abs() < 1.0e-3);
}

#[test]
fn update_mag_discards_ill_conditioned_calibration() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    *fusion.state.mag_calibrator = nearly_collinear_calibrator();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    let raw_mag = Vector3::new(10.0005, -4.9990, 3.00025);
    let mag_rub = frd_to_rub(raw_mag);

    fusion.integrate_mag(&mag_rub, 0);

    assert_eq!(fusion.state.consistency.sources.mag.innovation, 0.0);
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
