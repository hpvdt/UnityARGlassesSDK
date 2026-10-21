/*!
NaiveCF is the 6-DoF augmentation of the naive complementary filter: it
estimates the same 3 DoF attitude as `_3dof::NaiveCF` plus the 3 DoF
linear velocity, from the identical 9-axis IMU. It embeds the 3-DoF
estimator whole as its attitude channel (`NaiveCF::inner`), so both
estimators share not just the attitude machinery but the attitude
estimator itself.

# Accelerometer model

The accelerometer measures the specific force (felt acceleration), the
sum of the body's translational acceleration $a$ and the reaction to
gravity. With $S$ the body-to-world attitude and $G_{\mathrm{ACC}}$ the
felt gravity at rest $(0, 0, -9.81)$ m/s^2 in the FRD world frame, a level
reading decomposes as

$$f_{body} = S^{-1}\,(a_{world} + G_{\mathrm{ACC}})$$

so the world-frame translational acceleration follows by rotating the
reading through the attitude and removing felt gravity:

$$a_{world} = S\, f_{body} - G_{\mathrm{ACC}}$$

# Update channels

The attitude channel is the embedded 3-DoF estimator's per-sample update
(`update_acc_gyro`, `update_mag`): gyroscope dead-reckoning prediction,
then the gravity-direction state-sensor correction. That correction keeps
the `_3dof` gravity-only assumption — one 3-axis reading cannot
jointly constrain attitude and translational acceleration, so under a
steadily accelerating frame both estimators' level converges towards the
tilted specific force (the "1G artificial gravity" limitation recorded
in the fusion module docs; no complementary blend of the same single
reading can undo it, so the gravity channel keeps the
zero-acceleration assumption there).

The velocity channel is the 6-DoF addition, one complementary filter
following the shared update law
$S = \mathrm{ratio}\,(S^{-} + d\tilde{S}_1\, dt_1) + (1 - \mathrm{ratio})\, \tilde{S}_2$:

- rate sensor: $a_{world}$ above, the high-frequency dead-reckoning
  increment $a\, dt_v$;
- state sensor: the zero-velocity pseudo-observation $\tilde{S}_2 = 0$ —
  a head-mounted device is at rest far more often than it moves, so the
  correction is the auto-regression of the estimate towards zero;
- blend: rate-independent $\mathrm{ratio} = e^{-dt_v/\tau_v}$ with the
  regression time constant $\tau_v$, giving

$$v = e^{-dt_v/\tau_v}\,(v^{-} + a_{world}\, dt_v)$$

The regression bounds the accelerometer's integratable errors (bias,
attitude-induced gravity leakage) at $\tau_v$ times their magnitude,
and the estimate decays to zero whenever the device actually rests.
Acceleration bursts shorter than the gravity correction's lock time
(about $1 / (1 - \mathrm{ratio})$ samples) are tracked before the
level absorbs them as tilt. Under continuous rotation the gravity
correction carries a small level lag (the same lag the 3-DoF attitude
has), whose leaked gravity bounds the velocity accuracy on moving
traces — the measured floor is about $\|G_{\mathrm{ACC}}\| \cdot \tau_v$
times the sine of that lag.

# Relation to published filters

The attitude pipeline shared with `_3dof` is a
proportional-only instance of the explicit complementary filter of Mahony,
Hamel and Pflimlin, "Nonlinear
Complementary Filters on the Special Orthogonal Group" (IEEE Trans.
Autom. Control 53(5), 2008, DOI 10.1109/TAC.2008.923738): the gyroscope
propagates the attitude, and each state-sensor reading applies a
proportional correction along the rotation carrying the predicted
inertial direction onto the measured one. This repo keeps the
correction as a scaled rotation
(`naive_cf_core::get_correction`) with a fixed
per-sample blend fraction as the proportional gain, and leaves out the
integral (gyro-bias) term.

The 6-DoF pipeline — a complementary attitude filter feeding
gravity-compensated integration of the specific force in the world
frame, with zero-velocity information as the only low-frequency
velocity aid — is the foot-mounted pedestrian dead-reckoning
architecture of Fourati, Manamanni, Afilal and Handrich, "Position
Estimation Approach by Complementary Filter-aided IMU for Indoor
Environment" (ECC 2013; journal version "Heterogeneous Data Fusion
Algorithm for Pedestrian Navigation via Foot-Mounted Inertial
Measurement Unit and Complementary Filter", IEEE Trans. Instrum.
Meas. 64(1), 2015, DOI 10.1109/TIM.2014.2335912). The deviation: a
foot-mounted IMU has detectable stance phases, so Fourati resets the
velocity to zero at each detected stance (a hard zero-velocity
update), while head-mounted glasses have no reliable rest phase to
detect. This filter therefore applies the same zero-velocity
information continuously as the complementary blend above: an
exponentially forgetting (soft) zero-velocity update. The continuous
form cannot misfire on a false detection, at the price of never
fully trusting the pseudo-observation — a device under sustained
acceleration is pulled towards $v \approx a\,\tau_v$ instead of being
tracked indefinitely, and the estimate decays to zero over
$\tau_v$ whenever the device rests.
*/

use nalgebra::{UnitQuaternion, Vector3};

use crate::fusion::_3dof::naive_cf;
use crate::fusion::_3dof::naive_cf_core;
use crate::fusion::{rub_to_frd, Consistency, Fusion};
use crate::{any_glasses_or_dummy, ARGlasses, Error, GlassesEvent};

type Result<T> = std::result::Result<T, Error>;

/// Naive 6-DoF complementary-filter estimator.
///
/// Estimates the same 3-DoF attitude as the 3-DoF estimator plus the 3-DoF
/// world-frame linear velocity, from the identical 9-axis IMU: it embeds the
/// 3-DoF estimator whole and adds the velocity channel on top.
pub struct NaiveCF {
    /// Embedded 3-DoF attitude estimator: shared fusion state (device,
    /// attitude, dip, and calibration) and the whole attitude update law.
    pub inner: naive_cf::NaiveCF,
    /// Previous sample's translational acceleration estimate in the world
    /// FRD frame, m/s^2, and the previous AccGyro device timestamp, in
    /// microseconds. The vector is diagnostic state symmetric with
    /// `prev_gyro`'s; only the timestamp drives the velocity update, kept
    /// separately from the embedded estimator's `prev_gyro.1` so the bundled
    /// sensors stay independently consumable.
    pub prev_acc: (Vector3<f32>, u64), //FRD, world frame
}

impl NaiveCF {
    /// Time constant of the linear velocity's auto-regression to zero, in
    /// seconds: the complementary blend carrying the zero-velocity
    /// pseudo-observation, $1 - \mathrm{ratio} = 1 - e^{-dt_v/\tau_v}$. Sized
    /// to head-motion bursts ($\sim$1 s): long enough to track a burst
    /// without killing it, short enough to bound the integratable errors
    /// (level-wobble gravity leak, bias) it must regress away. Sustained
    /// velocity is not observable anyway — the gravity-only level lock
    /// absorbs it as tilt — so extending $\tau_v$ towards slower motion
    /// profiles would only raise the leak floor.
    pub(super) const VELOCITY_REGRESS_TAU_S: f32 = 2.0;

    /// Create a fusion instance over `glasses`, waiting for the first gyroscope reading.
    pub fn new(glasses: Box<dyn ARGlasses>) -> Result<Self> {
        // shared boot loop of the embedded 3-DoF estimator; the first
        // AccGyro sample's timestamp seeds the velocity channel's prev_acc
        let (inner, first_timestamp) = naive_cf::NaiveCF::new_with_first_sample(glasses)?;
        Ok(Self {
            inner,
            prev_acc: (Vector3::zeros(), first_timestamp),
        })
    }

    /// Create a 6-DoF fusion instance (attitude plus world-frame linear
    /// velocity) over whichever glasses are connected, or the SimMotion
    /// device: the 6-DoF counterpart of
    /// [`crate::fusion::_3dof::naive_cf::NaiveCF::any`].
    pub fn any() -> Result<Box<dyn Fusion>> {
        let glasses = any_glasses_or_dummy()?;
        Ok(Box::new(Self::new(glasses)?))
    }

    /// Linear velocity update: rotates the specific-force reading into the
    /// world FRD frame, removes felt gravity, integrates the translational
    /// acceleration over the elapsed AccGyro time, then applies the
    /// zero-velocity auto-regression. Runs even in free fall — the
    /// dead-reckoning prediction stays correct there, only the attitude's
    /// gravity correction is gated. Zero elapsed time (duplicate timestamps)
    /// skips integration while still refreshing `prev_acc`.
    pub(super) fn integrate_velocity(&mut self, acc_rub: &Vector3<f32>, t: u64) {
        let acc = rub_to_frd(acc_rub);

        // World-frame translational acceleration: the specific force
        // rotated through the current (already gravity-corrected) attitude,
        // with felt gravity removed.
        let a_world = self.inner.state.attitude * acc - naive_cf_core::G_ACC_FRD;

        let d_t = t - self.prev_acc.1;
        let d_t_s = d_t as f32 / naive_cf_core::GYRO_SPEED_IN_TIMESTAMP_FACTOR;
        if d_t_s > 0.0 {
            // Zero-velocity pseudo-observation blend: the regression towards
            // zero is the complementary state-sensor correction, with the
            // rate-independent ratio $e^{-dt_v/\tau_v}$.
            let regress = (-d_t_s / Self::VELOCITY_REGRESS_TAU_S).exp();
            self.inner.state.linear_velocity =
                (self.inner.state.linear_velocity + a_world * d_t_s) * regress;
        }

        self.prev_acc = (a_world, t);
    }
}

impl Fusion for NaiveCF {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        &mut self.inner.state.glasses
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        self.inner.state.attitude
    }

    fn linear_velocity(&self) -> Vector3<f32> {
        self.inner.state.linear_velocity
    }

    fn consistency(&self) -> Consistency {
        self.inner.state.consistency
    }

    fn update(&mut self) {
        let event = self.inner.state.next_event();
        match event {
            GlassesEvent::AccGyro {
                accelerometer,
                gyroscope,
                timestamp,
            } => {
                // the attitude channel is the embedded estimator's own
                // per-sample update; the velocity update runs last so it
                // consumes the reading through the attitude the gravity
                // correction just fixed
                self.inner
                    .update_acc_gyro(&accelerometer, &gyroscope, timestamp);
                self.integrate_velocity(&accelerometer, timestamp);
            }

            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                // The ACC-derived gravity hint keeps the calibrator's
                // preconditioned gravity surrogate anchored to the exact
                // magnetic dip, whose learned projection
                // $\kappa / (\gamma r)$ the result reports as `dip_sin`.
                // The field's vertical component refines the fusion
                // state's own `mag_dip_sin` estimate, and the attitude
                // update compares the full field against the dipped
                // reference built from it, so the magnetometer corrects
                // the heading and, jointly with the acc filter, the
                // level.
                self.inner.update_mag(&magnetometer, timestamp);
            }
            _ => {
                // TODO: handle KeyPress signal
            }
        }
    }
}
