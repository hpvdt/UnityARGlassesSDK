use nalgebra::{UnitQuaternion, Vector3};

use super::naive_cf::NaiveCF;
use super::test_fixture::{
    assert_level, felt_gravity_frd, frd_to_rub, run_sim_motion_scored, seeded_calibrator,
};
use crate::fusion::{AhrsCorrection, ConsistencyStatus, Fusion};

/// Closed form of the velocity recurrence
/// $v_n = (v_{n-1} + a\,dt_v)\,e^{-dt_v/\tau_v}$ for a constant
/// acceleration:
/// $v_n = e^{-n\,dt_v/\tau_v} v_0 + a\,dt_v\,e^{-dt_v/\tau_v}\,(1-e^{-n\,dt_v/\tau_v})/(1-e^{-dt_v/\tau_v})$.
fn expected_velocity(
    v0: Vector3<f64>,
    a: Vector3<f64>,
    dt_s: f64,
    tau_s: f64,
    n: u32,
) -> Vector3<f32> {
    let ratio = (-dt_s / tau_s).exp();
    let blend = dt_s * ratio * (1.0 - ratio.powi(n as i32)) / (1.0 - ratio);
    (v0 * ratio.powi(n as i32) + a * blend).map(|x| x as f32)
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

#[test]
fn update_mag_keeps_level_with_dipped_field() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // field dips 60 deg below the horizon and its horizontal component is
    // true north; the full-field correction carries level evidence too, but
    // the learned dip cancels the reference's 60 deg dip component, so the
    // level tilt stays bounded by the calibration fit residual (measured
    // 1.7e-4) — where a horizontal-only reference would instead tip the
    // attitude by the full unmodeled 60 deg dip
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));

    for t in 0..800 {
        fusion.integrate_mag(&dipped_rub, t);
        assert_level(&fusion.state.attitude, 5.0e-3);
    }
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
fn update_mag_refines_dip_to_self_consistent_pair() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.state.consistency.sources.mag = Default::default();

    // the dip state starts deliberately wrong (equator dip); the
    // per-sample complementary refinement pulls it towards the 60 deg
    // true dip the calibrated readings carry — but in this acc-less
    // loop nothing holds the level, so the seeded dip error trades
    // against the level along the level-dip unobservable direction and
    // the (level, dip) pair converges self-consistently (the acc filter
    // pins the level in production, letting the dip reach the true dip)
    let dip = 60.0f32.to_radians();
    let dipped_north = Vector3::new(dip.cos(), 0.0, dip.sin());
    let dipped_rub = frd_to_rub(offset + scale.component_mul(&dipped_north));
    fusion.state.mag_dip_sin = Some(0.0);

    // one step moves the state by exactly BASE_DIP_RATIO of the
    // residual, 0.05 * sin(60 deg) ~ 0.043; the fit's ~7.5 mrad
    // down-axis error is scaled down by the same factor to ~3.7e-4
    // (measured 3.9e-4), so a full-replacement or zero-blend update
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

    // the refinement keeps running: the dip climbs past half the seed
    // error (measured 58.0 deg against the 30 deg bound), so a state
    // that stops refining after the first step fails here
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!(
        dip_sin.asin() > 0.5 * dip,
        "dip stalled at {} deg",
        dip_sin.asin().to_degrees()
    );

    // the pair lands on the self-consistent manifold: the level tip
    // and the missing dip sum to the true dip (measured 59.6 deg; the
    // ~0.4 deg shortfall is the fixture's fit residual, so the same
    // 1 deg basis as the initialization test bounds it)
    let (_, pitch, _) = fusion.state.attitude.euler_angles();
    assert!(
        (pitch + dip_sin.asin() - dip).abs() < 1.0f32.to_radians(),
        "level/dip pair drifted off the manifold: pitch {} deg + dip {} deg vs {} deg",
        pitch.to_degrees(),
        dip_sin.asin().to_degrees(),
        dip.to_degrees()
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
    // turns away from the dipping field. The dip state initializes from
    // that same projection, so the (level, dip) pair starts exactly on
    // its self-consistent manifold and the full-field correction
    // vanishes there: the dip's fixed point stays the estimated-down
    // projection alone
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

#[test]
fn update_mag_locks_yaw_near_the_magnetic_poles() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let scale = Vector3::new(3.0, 2.0, 1.5);
    *fusion.state.mag_calibrator = seeded_calibrator(offset, scale);
    fusion.state.consistency.sources.mag = Default::default();

    // field ~0.6 deg off vertical: its 0.01 horizontal norm makes the
    // azimuth pure noise, but the full-field correction has no explicit
    // pole gate anymore — the near-vertical reference self-gates the
    // heading content to that vanishing horizontal component while the
    // tilt evidence still levels
    let near_vertical = Vector3::new(0.01, 0.0, (1.0_f32 - 1.0e-4).sqrt());
    fusion.state.attitude =
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2);
    let near_vertical_rub = frd_to_rub(offset + scale.component_mul(&near_vertical));

    let mut max_innovation = 0.0f32;
    for t in 0..800 {
        fusion.integrate_mag(&near_vertical_rub, t);
        max_innovation = max_innovation.max(fusion.state.consistency.sources.mag.innovation);
    }

    // the yaw stays locked: the residual drift is second order in the
    // tilt evidence (measured 5.6e-5), where an azimuth-trusting
    // correction would spin the heading toward the noise
    let (_, _, yaw) = fusion.state.attitude.euler_angles();
    assert!(
        (yaw - std::f32::consts::FRAC_PI_2).abs() < 1.0e-3,
        "yaw drifted to {yaw}"
    );

    // the level absorbs the tilt evidence: the correction tips by the
    // reference/reading gap (measured 0.008, the field's 0.01 horizontal
    // norm plus the fit residual), where a horizontal-only reference
    // would instead rotate by the ~1.55 rad north/vertical mismatch
    assert_level(&fusion.state.attitude, 0.05);
    assert!(
        max_innovation < 0.05,
        "pole innovation {max_innovation} exceeds the tilt evidence"
    );

    // the near-vertical reading still carries full dip evidence: the dip
    // state refined from it while the heading stayed locked (measured
    // 3e-5 against the exact-fit calibration's sub-mrad directional
    // error)
    let dip_sin = fusion.state.mag_dip_sin.expect("dip state present");
    assert!((dip_sin - near_vertical.z).abs() < 1.0e-3);
}

/// The 6-DoF estimator must level a stationary device exactly like the 3-DoF
/// one: this is the regression guard for gravity corrections that subtract a
/// per-sample acceleration estimate, which level only the first sample and
/// then freeze (a 30 deg tilt stays at 29.85 deg forever with such a design,
/// while the plain gravity correction converges to zero).
#[test]
fn gravity_correction_levels_a_stationary_device() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::from_euler_angles(0.0, 30.0f32.to_radians(), 0.0);
    fusion.prev_gyro = (Vector3::zeros(), 0);
    fusion.prev_acc = (Vector3::zeros(), 0);
    fusion.state.consistency.sources.acc = Default::default();

    // the device is level and at rest: the reading is felt gravity, which
    // the 30 deg-pitched estimate mispredicts in body frame
    let acc_rub = frd_to_rub(felt_gravity_frd());

    const DT_US: u64 = 40_000;
    for sample in 1..=1500u32 {
        let t = sample as u64 * DT_US;
        fusion.integrate_gyro(&frd_to_rub(Vector3::zeros()), t);
        fusion.integrate_acc(&acc_rub, t);
        fusion.integrate_velocity(&acc_rub, t);
        if sample == 250 {
            // the blend weight $1 - \mathrm{ratio}$ (`BASE_GRAV_RATIO`)
            // leaves the retention factor $\mathrm{ratio}^{250}$ scaling
            // 30 deg down to ~8.5 deg;
            // a frozen design sits at ~29.9 deg here
            assert!(
                fusion.state.attitude.angle() < 15.0f32.to_radians(),
                "tilt stalled at {} deg after 250 samples",
                fusion.state.attitude.angle().to_degrees()
            );
        }
    }

    assert!(
        fusion.state.attitude.angle() < 0.5f32.to_radians(),
        "residual tilt after 60 s: {}",
        fusion.state.attitude.angle()
    );
    // leveling a rest reading leaves the velocity at zero: the early tilt
    // leaks bounded spurious velocity, but both the tilt and the leak decay
    // long before the run ends
    assert!(
        fusion.state.linear_velocity.norm() < 0.02,
        "residual velocity after leveling: {}",
        fusion.state.linear_velocity.transpose()
    );
}

#[test]
fn velocity_stays_zero_at_rest() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_acc = (Vector3::zeros(), 0);

    // level rest: the reading is felt gravity exactly, so the world-frame
    // translational acceleration is zero and so is the velocity
    let acc_rub = frd_to_rub(felt_gravity_frd());
    for sample in 1..=100u64 {
        fusion.integrate_velocity(&acc_rub, sample * 40_000);
    }

    assert!(
        fusion.state.linear_velocity.norm() < 1.0e-6,
        "rest velocity {}",
        fusion.state.linear_velocity.transpose()
    );
}

#[test]
fn velocity_regresses_to_zero() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_acc = (Vector3::zeros(), 0);
    let v0 = Vector3::new(0.5, -0.3, 0.2);
    fusion.state.linear_velocity = v0;

    // 250 samples of 40 ms: the auto-regression is the exact per-sample
    // factor $e^{-dt_v/\tau_v}$, so 10 s decays the initial velocity by
    // $e^{-5}$
    let acc_rub = frd_to_rub(felt_gravity_frd());
    for sample in 1..=250u64 {
        fusion.integrate_velocity(&acc_rub, sample * 40_000);
    }

    let expected = v0 * (-(10.0f32 / NaiveCF::VELOCITY_REGRESS_TAU_S)).exp();
    assert!((fusion.state.linear_velocity - expected).norm() < 1.0e-5);
}

#[test]
fn velocity_integrates_constant_acceleration() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_acc = (Vector3::zeros(), 0);

    // identity attitude: body frame equals world frame, and the reading is
    // the sustained acceleration plus felt gravity
    let a_world = Vector3::new(2.0, -1.0, 0.5);
    let acc_rub = frd_to_rub(a_world + felt_gravity_frd());
    const DT_S: f64 = 0.04;
    for sample in 1..=250u32 {
        fusion.integrate_velocity(&acc_rub, sample as u64 * 40_000);
    }

    let expected = expected_velocity(
        Vector3::new(0.0, 0.0, 0.0),
        a_world.map(|x| x as f64),
        DT_S,
        NaiveCF::VELOCITY_REGRESS_TAU_S as f64,
        250,
    );
    assert!(
        (fusion.state.linear_velocity - expected).norm() < 1.0e-3,
        "estimated {} vs expected {}",
        fusion.state.linear_velocity.transpose(),
        expected.transpose()
    );
}

#[test]
fn velocity_tracks_short_acceleration_burst() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_gyro = (Vector3::zeros(), 0);
    fusion.prev_acc = (Vector3::zeros(), 0);

    // 2 m/s^2 for 2 s (50 samples), then rest: the burst is far shorter
    // than the gravity correction's lock time (about $1/(1-\mathrm{ratio})$
    // samples), so the level stays near true and the velocity tracks the
    // burst; the
    // acceleration is not truly sustained, so no artificial-gravity lock
    // forms and afterwards the estimate regresses back to zero
    let burst_rub = frd_to_rub(Vector3::new(2.0, 0.0, 0.0) + felt_gravity_frd());
    let rest_rub = frd_to_rub(felt_gravity_frd());
    const DT_US: u64 = 40_000;

    for sample in 1..=50u64 {
        let t = sample * DT_US;
        fusion.integrate_gyro(&frd_to_rub(Vector3::zeros()), t);
        fusion.integrate_acc(&burst_rub, t);
        fusion.integrate_velocity(&burst_rub, t);
    }
    let burst_end = fusion.state.linear_velocity.x;
    // regression alone would reach the closed-form
    // $a\, dt_v\, \mathrm{ratio}\, (1 - \mathrm{ratio}^{50}) / (1 - \mathrm{ratio})$
    // ~ 2.3 m/s; the early level lock costs a fraction of it
    assert!(
        (1.4..=2.7).contains(&burst_end),
        "burst velocity {burst_end}"
    );

    for sample in 51..=1050u64 {
        let t = sample * DT_US;
        fusion.integrate_gyro(&frd_to_rub(Vector3::zeros()), t);
        fusion.integrate_acc(&rest_rub, t);
        fusion.integrate_velocity(&rest_rub, t);
    }
    // 20 s after the burst the residual tilt (at most the burst-end ~2.5
    // deg, itself re-leveling) leaks bounded spurious velocity
    let settled = fusion.state.linear_velocity.norm();
    assert!(settled < 1.0, "post-burst velocity {settled}");

    for sample in 1051..=1550u64 {
        let t = sample * DT_US;
        fusion.integrate_gyro(&frd_to_rub(Vector3::zeros()), t);
        fusion.integrate_acc(&rest_rub, t);
        fusion.integrate_velocity(&rest_rub, t);
    }
    let regressed = fusion.state.linear_velocity.norm();
    assert!(regressed < 0.05, "final velocity {regressed}");
}

#[test]
fn free_fall_integrates_downward_without_attitude_correction() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_acc = (Vector3::zeros(), 0);
    fusion.state.consistency.sources.acc = Default::default();

    // zero specific force: below the 1 m/s^2 gravity-correction gate, so
    // the attitude must not move, but the velocity channel dead-reckons
    // the free fall (a_world = -G_ACC_FRD, accelerating downward)
    let acc_rub = frd_to_rub(Vector3::zeros());
    for sample in 1..=50u32 {
        fusion.integrate_velocity(&acc_rub, sample as u64 * 40_000);
    }

    assert_eq!(fusion.state.consistency.sources.acc.innovation, 0.0);
    assert_eq!(fusion.state.attitude.angle(), 0.0);
    let expected = expected_velocity(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 9.81),
        0.04,
        NaiveCF::VELOCITY_REGRESS_TAU_S as f64,
        50,
    );
    assert!((fusion.state.linear_velocity - expected).norm() < 1.0e-3);
    assert!(fusion.state.linear_velocity.z > 0.0); // +z is down in FRD
}

#[test]
fn duplicate_timestamp_skips_velocity_update() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    fusion.state.attitude = UnitQuaternion::identity();
    fusion.prev_acc = (Vector3::zeros(), 0);

    let acc_rub = frd_to_rub(Vector3::new(1.0, 0.0, 0.0) + felt_gravity_frd());
    fusion.integrate_velocity(&acc_rub, 100_000);
    let after_first = fusion.state.linear_velocity;

    // duplicate timestamp: zero elapsed time must not integrate or regress
    fusion.integrate_velocity(&acc_rub, 100_000);

    assert!((fusion.state.linear_velocity - after_first).norm() < 1.0e-9);
    assert_eq!(fusion.prev_acc.1, 100_000);
}

#[test]
fn fusion_trait_exposes_linear_velocity() {
    let mut fusion = NaiveCF::new(Box::new(crate::sim::SimMotion::new())).unwrap();
    let velocity = Vector3::new(0.3, -0.2, 0.1);
    fusion.state.linear_velocity = velocity;

    assert_eq!(Fusion::linear_velocity(&fusion), velocity);

    // the wrapper forwards the world-FRD state untouched: its attitude
    // output conventions do not apply to linear velocity
    let ahrs = AhrsCorrection::frd(Box::new(fusion));
    assert_eq!(Fusion::linear_velocity(&ahrs), velocity);
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
/// cleanly. With it boxed, the measured debug peak is 256-320 KiB.
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

/// End-to-end SimMotion run on the default continuous-motion profile: the
/// 20 RPM rotation leaves the gravity correction a ~2 deg lag (the
/// pre-existing 3-DoF behavior this estimator shares), whose leaked
/// gravity dominates the velocity error — the documented unobservability
/// of this estimator family, so this test pins the honest envelope: the
/// velocity stays bounded, the error does not accumulate across the scored
/// halves, and the 9-axis innovation report stays healthy.
#[test]
fn velocity_stays_bounded_on_continuous_motion() {
    /// Virtual events skipped before scoring: attitude convergence plus
    /// velocity time constants.
    const WARMUP_EVENTS: usize = 1500;
    /// Scored span, split into halves for the no-accumulation check.
    const SCORED_EVENTS: usize = 3000;
    /// Bound on the mean estimated speed over the scored span, in m/s
    /// (measured 0.63: the rotation-lag leak floor at $\tau_v$ gain).
    const MAX_MEAN_ESTIMATE_NORM: f32 = 2.0;
    /// Margin by which the second half's mean error may exceed the first
    /// half's, in m/s: the auto-regression must hold the leak floor flat
    /// (measured 0.76 -> 0.55).
    const HALVES_ERROR_MARGIN: f32 = 0.15;

    let run = run_sim_motion_scored(
        Default::default(),
        WARMUP_EVENTS,
        SCORED_EVENTS,
        |glasses| Box::new(NaiveCF::new(glasses).unwrap()),
    );

    /// Bound on the mean attitude angle error over the scored span, in rad
    /// (measured 0.057: the rotation-lag floor at ~3.2 deg, the baseline
    /// the EKF benchmark compares against).
    const MAX_MEAN_ATTITUDE_ERROR: f32 = 0.065;

    assert!(
        run.mean_estimate_norm < MAX_MEAN_ESTIMATE_NORM,
        "mean |v_est| = {}",
        run.mean_estimate_norm
    );
    assert!(
        run.mean_error_second < run.mean_error_first + HALVES_ERROR_MARGIN,
        "error accumulated: {} -> {}",
        run.mean_error_first,
        run.mean_error_second
    );
    assert!(
        run.mean_attitude_error < MAX_MEAN_ATTITUDE_ERROR,
        "mean attitude error = {}",
        run.mean_attitude_error
    );
    assert_eq!(
        run.consistency.sources.acc.status(),
        ConsistencyStatus::Consistent
    );
    assert_eq!(
        run.consistency.sources.gyro.status(),
        ConsistencyStatus::Consistent
    );
}

/// End-to-end SimMotion run on a slow-rotation gentle-motion profile: the
/// attitude lag (and with it the leaked-gravity velocity floor) shrinks
/// with the rotation rate, so the velocity error settles near the leak
/// residual and the estimate keeps the true speed scale. Wiring the
/// learned dip into the attitude correction adds a cold-start phase on
/// top: while the mag calibration converges, the full-field correction
/// chases the calibrator's azimuthal error, and the level content of that
/// chase — fought by the acc filter — leaks gravity into the velocity.
/// The re-tuned `BASE_MAG_RATIO`/`BASE_DIP_RATIO` keep that leak at the
/// settled attitude-lag floor (see the constants' doc comments for the
/// measured heading-lock trade the re-tune makes), so both scored halves
/// pin the same floor and the cold-start bound doubles as the regression
/// guard against the transient re-inflating. Instantaneous direction is
/// leak-dominated and not asserted; the clean tracking evidence is the
/// synthetic closed-form, burst, and free-fall tests.
#[test]
fn velocity_tracks_gentle_motion() {
    const WARMUP_EVENTS: usize = 1500;
    const SCORED_EVENTS: usize = 3000;
    /// Bound on the cold-start half's mean absolute velocity error, in
    /// m/s (measured 0.38): the calibration-transient leak must stay at
    /// the settled floor's scale — at the pre-re-tune blend ratios this
    /// half measured 1.54 m/s, so a regressed cold-start transient fails
    /// here well before the settled bound would.
    const MAX_COLD_START_ABS_ERROR: f32 = 0.45;
    /// Bound on the settled half's mean absolute velocity error, in m/s
    /// (measured 0.29, the pre-wiring floor): after calibration, dip,
    /// and level converge, only the attitude-lag leak may remain.
    const MAX_SETTLED_ABS_ERROR: f32 = 0.45;
    /// Ratio band for mean estimated vs true speed (measured 1.25): the
    /// estimate must keep the true speed scale — neither the collapsed
    /// nor the leak-inflated side of the band is acceptable.
    const MAGNITUDE_RATIO_BAND: (f32, f32) = (0.6, 2.0);

    let config = crate::sim::sim_motion::Config {
        seed: 0xA15E_D00D_5EED_0002,
        max_body_rate_rpm: 3.0,
        linear_jerk_std_dev: 0.8,
        linear_accel_damping: 0.8,
        max_linear_accel: 1.5,
        linear_velocity_damping: 0.4,
        max_linear_velocity: 1.5,
        ..Default::default()
    };
    let run = run_sim_motion_scored(config, WARMUP_EVENTS, SCORED_EVENTS, |glasses| {
        Box::new(NaiveCF::new(glasses).unwrap())
    });

    /// Bound on the mean attitude angle error over the scored span, in rad
    /// (measured 0.097, ~5.6 deg: the calibration-transient attitude error
    /// dominating this profile's scored span, the baseline the EKF
    /// benchmark compares against).
    const MAX_MEAN_ATTITUDE_ERROR: f32 = 0.11;

    assert!(
        run.mean_error_first < MAX_COLD_START_ABS_ERROR,
        "cold-start |v_est - v_true| = {}",
        run.mean_error_first
    );
    assert!(
        run.mean_error_second < MAX_SETTLED_ABS_ERROR,
        "settled |v_est - v_true| = {}",
        run.mean_error_second
    );
    let magnitude_ratio = run.mean_estimate_norm / run.mean_truth_norm;
    assert!(
        (MAGNITUDE_RATIO_BAND.0..MAGNITUDE_RATIO_BAND.1).contains(&magnitude_ratio),
        "speed scale drifted: |v_est| {} vs |v_true| {}",
        run.mean_estimate_norm,
        run.mean_truth_norm
    );
    assert!(
        run.mean_attitude_error < MAX_MEAN_ATTITUDE_ERROR,
        "mean attitude error = {}",
        run.mean_attitude_error
    );
}
