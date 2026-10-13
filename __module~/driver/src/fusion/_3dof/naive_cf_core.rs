/*!
Shared attitude pipeline of the naive complementary filters.

The estimators built on [`FusionState`] run the same attitude machinery:
gyroscope dead-reckoning, the gravity-direction correction helper, the
magnetometer attitude correction, the dip-state refinement, attitude
renormalization, and the blocking device event loop. This module holds
that machinery once, together with the shared blend-ratio and frame
constants. Each estimator owns its state struct and calls these free
functions with `&mut FusionState` plus its own previous-sample
bookkeeping (e.g. `prev_gyro`), so estimators delegate through thin
methods without sharing a struct layout.

All estimated state follows the FRD world-frame conventions of the fusion
module; raw sensor arguments stay in the device's RUB frame and are
converted at the boundary.
*/

use nalgebra::{UnitQuaternion, Vector3};

use crate::fusion::{rub_to_frd, FusionState};
use crate::GlassesEvent;

//CAUTION: right-multiplication means rotation, unconventionally
//(scaled_rotation_between(pred, meas) paired with attitude = attitude * correction
// moves the prediction towards the measurement)

pub(super) const BASE_GRAV_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.005;
//const BASE_GRAV_RATIO: f32 = 0.0; //no grav
// const BASE_GRAV_RATIO: f32 = 1.0; //absolute correction, no gyro

/// Blend ratio $1 - \mathrm{ratio}$ of the magnetometer attitude
/// correction of [`integrate_mag`], tuned for the dipped full-field
/// update: unlike the pre-wiring horizontal-only partition, the
/// shortest arc between the estimated and measured field carries level
/// evidence, and while the mag calibration converges that arc is driven
/// by the calibrator's azimuthal error — the chase tips the level by
/// $\sin\delta$ of the azimuthal residual per sample, and the acc filter
/// pulls it back at `BASE_GRAV_RATIO`, leaking the tilt into the 6-DoF
/// velocity at $\tau_v$ gain. The cold-start level tilt therefore
/// settles at the mag/grav balance of that drive: on the gentle-motion
/// SimMotion profile the cold-start scored half's mean velocity error
/// measures 1.54 m/s at the pre-wiring 0.005, 0.81 at 0.0025, and 0.38
/// at this 0.00125 — the settled floor's scale. The heading lock time
/// trades against it with $1/\mathrm{ratio}$ scaling: a 30 deg heading
/// step decays with a measured ~33 s e-fold (against ~9 s at 0.005) to
/// the same ~1 deg converged floor, and the default 20 RPM profile,
/// whose velocity error is the rotation-lag leak, is unaffected
/// (measured mean |v_est| 0.63 m/s, halves 0.76 -> 0.55 m/s).
pub(super) const BASE_MAG_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.00125;

/// Blend ratio $1 - \mathrm{ratio}$ of the per-sample dip-state update. The
/// vertical field component feeds [`FusionState::mag_dip_sin`] rather
/// than the attitude correction directly, so its innovation is
/// yaw-invariant and the remaining disturbance is the roll/pitch error
/// the acc filter keeps small; the ratio can therefore stay small
/// relative to the reading noise, tracking the slow evolution of the
/// field's vertical component — the geographic drift of the true dip
/// and, during the mag calibration's cold start, the converging
/// calibrator's vertical error, which the dip state must absorb before
/// the attitude correction can tip the level with it (2.5x the
/// pre-re-tune 0.02: measured gentle-motion cold-start mean velocity
/// error 0.42 -> 0.38 m/s alongside the `BASE_MAG_RATIO` re-tune).
///
/// `pub(in crate::fusion)` rather than `pub(super)`: both estimators' test
/// suites pin the blend rate against this constant.
pub(in crate::fusion) const BASE_DIP_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.05;

pub(in crate::fusion) const GYRO_SPEED_IN_TIMESTAMP_FACTOR: f32 = 1000.0 * 1000.0; // microseconds

/// Felt gravity (specific force at rest) in the FRD world frame, m/s^2.
pub(in crate::fusion) const G_ACC_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);
/// Unit down axis of the FRD world frame.
pub(super) const DOWN_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, 1.0);

/// Reads until the next valid event. Blocks.
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
) {
    let gyro = rub_to_frd(gyro_rub);

    let d_t1 /*$dt_1$*/ = t - prev_gyro.1;
    let d_t1_f = d_t1 as f32 / GYRO_SPEED_IN_TIMESTAMP_FACTOR;
    let d_s1_t1 /*$d\tilde{S}_1\, dt_1$*/ = d_t1_f * gyro;

    if d_t1_f > 0.0 {
        let increment = UnitQuaternion::from_euler_angles(d_s1_t1.x, d_s1_t1.y, d_s1_t1.z);

        state.attitude *= increment;
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
pub(in crate::fusion) fn integrate_acc(state: &mut FusionState, acc_rub: &Vector3<f32>, _t: u64) {
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

/// Magnetometer update, with the magnetic dip $\delta$ wired into the
/// attitude correction: the FRD reference field is
/// $(\cos\delta, 0, \sin\delta)$, positive below the horizon, built from
/// the learned dip state, so the reading is never corrected towards a
/// purely horizontal north. The single vector observation cannot fix
/// attitude and dip jointly — a level (roll/pitch) error and a dip error
/// produce the same vertical residual — so the update is partitioned in
/// time, not in direction: the field's vertical component, measured
/// through the acc-maintained level, refines only
/// [`FusionState::mag_dip_sin`] (initialized from the first usable
/// reading, independently of the calibrator's `dip_sin` fit statistic),
/// while the attitude update compares the full calibrated field against
/// the dipped reference, feeding both the heading and the level. Both
/// steps share the same gating condition — the calibrator's usable
/// calibrated direction — so a field near the magnetic poles, whose
/// azimuth is noise, self-gates its heading content to the vanishing
/// horizontal component while its dip evidence still moves the level.
pub(in crate::fusion) fn integrate_mag(state: &mut FusionState, mag_rub: &Vector3<f32>, t: u64) {
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
    }
    .clamp(-1.0, 1.0);
    state.mag_dip_sin = Some(dip_sin);

    // Attitude step: the full calibrated field `mag_normalised` is
    // compared against the estimated dipped reference field
    // `estimated_field` = the world reference $(\cos\delta, 0, \sin\delta)$
    // of the just-refined `dip_sin`, rotated into the body frame, so the
    // correction carries both the heading and the level evidence the
    // reading holds. CAUTION: a level (roll/pitch) error and a dip error
    // produce the same vertical residual, so this correction and the dip
    // refinement above converge to a self-consistent (level, dip) pair
    // rather than the true level alone — the acc filter keeps the level
    // observable in production, and the jointly tuned `BASE_MAG_RATIO`
    // and `BASE_DIP_RATIO` (see their doc comments for the measured
    // heading-lock/cold-start trade) keep the correction's level pull
    // small enough that the calibrator's converging directional error
    // cannot tilt it through the acc filter. The reference's unit norm
    // makes the near-pole azimuth gate unnecessary: a field approaching
    // the vertical carries its heading evidence in a vanishing
    // horizontal component, while its (now usable) dip evidence still
    // levels.
    let dip_cos = (1.0 - dip_sin * dip_sin).sqrt();
    let estimated_field = attitude.inverse() * Vector3::new(dip_cos, 0.0, dip_sin);
    let correction_opt =
        UnitQuaternion::scaled_rotation_between(&estimated_field, &mag_normalised, BASE_MAG_RATIO);

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
