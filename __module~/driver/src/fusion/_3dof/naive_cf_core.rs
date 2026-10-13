//!
//! Shared attitude pipeline of the naive complementary filters.
//!
//! The estimators built on [`FusionState`] run the same attitude machinery:
//! gyroscope dead-reckoning, the gravity-direction correction helper, the
//! magnetometer heading correction, the dip-state refinement, attitude
//! renormalization, and the blocking device event loop. This module holds
//! that machinery once, together with the shared blend-ratio and frame
//! constants. Each estimator owns its state struct and calls these free
//! functions with `&mut FusionState` plus its own previous-sample
//! bookkeeping (e.g. `prev_gyro`), so estimators delegate through thin
//! methods without sharing a struct layout.
//!
//! All estimated state follows the FRD world-frame conventions of the fusion
//! module; raw sensor arguments stay in the device's RUB frame and are
//! converted at the boundary.
//!

use nalgebra::{UnitQuaternion, Vector3};

use crate::fusion::{rub_to_frd, FusionState};
use crate::GlassesEvent;

//CAUTION: right-multiplication means rotation, unconventionally
//(scaled_rotation_between(pred, meas) paired with attitude = attitude * correction
// moves the prediction towards the measurement)

pub(super) const BASE_GRAV_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.005;
//const BASE_GRAV_RATIO: f32 = 0.0; //no grav
// const BASE_GRAV_RATIO: f32 = 1.0; //absolute correction, no gyro

pub(super) const BASE_MAG_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.005;

/// Blend ratio $1 - \mathrm{ratio}$ of the per-sample dip-state update. The
/// vertical field component feeds [`FusionState::mag_dip_sin`] instead of
/// the attitude, so its innovation is yaw-invariant and the remaining
/// disturbance is the roll/pitch error the acc filter keeps small; the
/// ratio can therefore be small, tracking the slow geographic drift of
/// the true dip rather than chasing single noisy readings.
pub(super) const BASE_DIP_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.02;

/// Minimum norm of the calibrated field's horizontal component, as a
/// fraction of the unit field, below which the heading update is gated.
/// A field closer than $\arcsin(0.1) \approx 5.7^{\circ}$ to vertical
/// (near the magnetic poles) carries almost no azimuth information, so
/// its horizontal noise would rotate the attitude about the down axis
/// without actual heading evidence.
pub(super) const MIN_HEADING_MAG_FIELD_NORM: f32 = 0.1;

pub(in crate::fusion) const GYRO_SPEED_IN_TIMESTAMP_FACTOR: f32 = 1000.0 * 1000.0; //microseconds

/// Felt gravity (specific force at rest) in the FRD world frame, m/s^2.
pub(in crate::fusion) const G_ACC_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);
/// Unit down axis of the FRD world frame.
pub(super) const DOWN_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, 1.0);
/// Unit magnetic-north axis of the FRD world frame; the field's
/// horizontal component points here whatever the dip angle.
pub(super) const NORTH_FRD: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);

///read until next valid event. Blocks.
pub(in crate::fusion) fn next_event(state: &mut FusionState) -> GlassesEvent {
    loop {
        match state.glasses.read_event() {
            Ok(event) => return event,
            Err(e) => {
                println!("Error reading event: {}", e);
            }
        }
    }
}

/// Dead-reckoning prediction step of the complementary filter: applies the
/// gyroscope increment $d\tilde{S}_1\, dt_1$ to the attitude before any
/// state-sensor correction. `gyro_rub` is a RUB angular rate in rad/s, `t`
/// a device timestamp in microseconds, and the applied increment magnitude
/// is recorded into the gyro consistency source as the sensor-health
/// signal. Zero elapsed time (duplicate timestamps) skips propagation
/// and recording while still refreshing `prev_gyro`.
pub(in crate::fusion) fn integrate_gyro(
    state: &mut FusionState,
    prev_gyro: &mut (Vector3<f32>, u64),
    gyro_rub: &Vector3<f32>,
    t: u64,
) -> () {
    let gyro = rub_to_frd(gyro_rub);

    let d_t1 /*$dt_1$*/ = t - prev_gyro.1;
    let d_t1_f = d_t1 as f32 / GYRO_SPEED_IN_TIMESTAMP_FACTOR;
    let d_s1_t1 /*$d\tilde{S}_1\, dt_1$*/ = d_t1_f * gyro;

    if d_t1_f > 0.0 {
        let increment = UnitQuaternion::from_euler_angles(d_s1_t1.x, d_s1_t1.y, d_s1_t1.z);

        state.attitude = state.attitude * increment;
        let _ = state.consistency.sources.gyro.record(increment.angle());
    }

    *prev_gyro = (gyro, t);
}

/// Scaled rotation correcting the estimated felt-gravity direction
/// (`rotation * G_ACC_FRD`) towards the measured direction `acc`, applying a
/// `scale` fraction ($1 - \mathrm{ratio}$) of the full residual angle.
/// Returns `None` when the two directions are opposite, in which case the
/// caller has no correction to apply.
pub(in crate::fusion) fn get_correction(
    acc: &Vector3<f32>,
    rotation: &UnitQuaternion<f32>,
    scale /*$1 - \mathrm{ratio}$*/: f32,
) -> Option<UnitQuaternion<f32>> {
    let uncorrected = rotation * G_ACC_FRD.normalize();

    UnitQuaternion::scaled_rotation_between(&uncorrected, &acc.normalize(), scale)
}

/// Gravity-direction state-sensor correction: treats the accelerometer
/// reading as pure felt gravity, the assumption both naive estimators'
/// attitude channels share. Readings weaker than 1 m/s^2 (free fall, or the
/// sensor disabled) carry no usable direction and are skipped. The applied
/// correction is recorded into the acc consistency source, reconstructed
/// from the post-blend angle through the blend ratio.
pub(in crate::fusion) fn integrate_acc(
    state: &mut FusionState,
    acc_rub: &Vector3<f32>,
    _t: u64,
) -> () {
    let acc = rub_to_frd(acc_rub);

    if acc.norm() < 1.0 {
        return; //almost in free fall, or acc disabled, do not correct
    }

    let attitude = &state.attitude;
    let correction_opt = get_correction(&acc, &attitude.inverse(), BASE_GRAV_RATIO);

    match correction_opt {
        Some(correction_inv) => {
            let correction = correction_inv.inverse();
            let _ = state
                .consistency
                .sources
                .acc
                .record_scaled(correction.angle(), BASE_GRAV_RATIO);

            state.attitude = attitude * correction;
        }
        None => {
            // opposite direction, don't know how to correct
        }
    }
}

/// Magnetometer update with the magnetic dip $\delta$ partitioned out
/// of the heading correction: the FRD reference field is
/// $(\cos\delta, 0, \sin\delta)$, positive below the horizon, so the
/// reading is never corrected towards a purely horizontal north. The
/// single vector observation cannot fix attitude and dip jointly — a
/// level (roll/pitch) error and a dip error produce the same vertical
/// residual — so the update is partitioned: the field's vertical
/// component, measured through the acc-maintained level, refines only
/// [`FusionState::mag_dip_sin`] (initialized from the first usable
/// reading, independently of the calibrator's `dip_sin` fit statistic),
/// while the attitude update compares only the field's horizontal
/// component against estimated north (leaving level to the acc filter).
pub(in crate::fusion) fn integrate_mag(
    state: &mut FusionState,
    mag_rub: &Vector3<f32>,
    t: u64,
) -> () {
    let mag_raw = rub_to_frd(mag_rub); // reading is always muT (microTesla) pointing to north

    // gravity direction is already estimated by the acc complementary filter
    let gravity_hint: Option<Vector3<f32>> = Some(state.attitude.inverse() * G_ACC_FRD);
    let result = match state
        .mag_calibrator
        .evaluate_correct(mag_raw, gravity_hint, t)
    {
        Ok(result) => result,
        Err(_) => return,
    };
    // `direction` is already the normalized calibrated unit vector.
    let mag_normalised: Vector3<f32> = match result.direction {
        Some(direction) => direction,
        None => return,
    };

    // Dip step: the vertical component of the calibrated field, measured
    // through the level the acc filter maintains, is the dip evidence the
    // fusion state owns. It is invariant to heading error (a rotation
    // about the down axis preserves it), so even the large yaw transient
    // right after boot cannot flip the update's sign; only roll/pitch
    // error propagates, which the acc filter holds small relative to
    // `BASE_DIP_RATIO`'s pull. The first usable reading initializes the
    // state directly — the field's vertical projection onto the
    // estimated down axis — so it exists even while the calibrator's
    // gravity surrogate is unseeded and its `dip_sin` record absent, and
    // it never returns to `None` afterwards.
    let attitude = &state.attitude;
    let down_body = attitude.inverse() * DOWN_FRD;
    let vertical = mag_normalised.dot(&down_body);
    let dip_sin = match state.mag_dip_sin {
        Some(previous) => previous + BASE_DIP_RATIO * (vertical - previous),
        None => vertical,
    };
    state.mag_dip_sin = Some(dip_sin.clamp(-1.0, 1.0));

    // Heading step: only the field's horizontal component (a pure
    // north reading) is compared against estimated north. Both vectors
    // are perpendicular to estimated down — north as a world
    // horizontal, the measured component by construction — so the
    // correction is a pure heading rotation about the down axis and
    // can never tip the level the acc filter maintains, whatever the
    // field's dip. The vertical component is invariant to heading
    // error (a rotation about the down axis preserves it) and carries
    // exactly the dip the heading correction must stay blind to.
    let mag_horizontal = mag_normalised - vertical * down_body;
    if mag_horizontal.norm() < MIN_HEADING_MAG_FIELD_NORM {
        return;
    }

    let estimated_north = attitude.inverse() * NORTH_FRD;
    let correction_opt =
        UnitQuaternion::scaled_rotation_between(&estimated_north, &mag_horizontal, BASE_MAG_RATIO);

    match correction_opt {
        Some(correction_inv) => {
            let correction = correction_inv.inverse();
            let _ = state
                .consistency
                .sources
                .mag
                .record_scaled(correction.angle(), BASE_MAG_RATIO);
            state.attitude = attitude * correction;
        }
        None => {
            // opposite direction, don't know how to correct
        }
    }
}

/// Renormalizes the attitude quaternion to counter numerical drift.
pub(in crate::fusion) fn renormalize(state: &mut FusionState) {
    // state.attitude.renormalize_fast(); // TODO: switch to it after rigorous testing
    state.attitude.renormalize();
}
