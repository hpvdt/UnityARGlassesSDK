/*!
NaiveCF is a simple sensor fusion algorithm that uses a complementary filter.

Complementary filter with a very simple update algorithm. With $S$ and $S^{-}$ the current and
previous estimated state, $d\tilde{S}_1$ a rate sensor reading (e.g. gyroscope: high frequency, high
drift, dead reckoning), $\tilde{S}_2$ a state sensor reading (e.g. gravity/accelerometer, magnetometer:
low frequency, high noise, low drift), $dt_1$ the time elapsed since the last rate sensor sample, and
$\mathrm{ratio}$ the blend ratio:

$$S = \mathrm{ratio}\, (S^{-} + d\tilde{S}_1\, dt_1) + (1 - \mathrm{ratio})\, \tilde{S}_2$$

This implies:

- The algorithm natively supports state sensor(s) with different sampling frequency, incomplete
  reading, or unreliable reading, as the $\tilde{S}_2$ variable is merely an optional correction.
- The interpolation between $\tilde{S}_2$ and the rate-integrated prediction doesn't need to be linear
  or additive, e.g. 3D angular interpolation is multiplicative.
- The ratio can be adjusted based on quality and frequency of the state sensor(s).

Most glasses have acc & grav/acc readings in 1 bundle, but I prefer not using this assumption and still
update them independently.

The state-sensor half of the attitude pipeline shared with the other naive
complementary-filter estimators (the magnetometer attitude correction and
the gravity-direction correction) lives in `naive_cf_core` as methods of the
shared `FusionState`; the algorithm-agnostic state services (the device
event loop and renormalization) live on the fusion module's own
`FusionState` impl. This estimator owns
the rate-sensor half — the gyroscope dead-reckoning prediction, which
consumes its `prev_gyro` bookkeeping — and the per-sample update law
(`update_acc_gyro`, `update_mag`) the 6-DoF estimator embeds whole as its
attitude channel. The gravity correction keeps the 3-DoF-specific assumption
that reads the accelerometer as a pure gravity-direction sensor.

# example:
 TODO: fill
*/

use nalgebra::{UnitQuaternion, Vector3};

use super::naive_cf_core;
use crate::fusion::{rub_to_frd, Consistency, Fusion, FusionState};
use crate::{any_glasses_or_dummy, ARGlasses, Error, GlassesEvent};

type Result<T> = std::result::Result<T, Error>;

/// Naive 3-DoF complementary-filter attitude estimator.
///
/// The attitude comes from gyroscope dead-reckoning corrected by the
/// accelerometer (level) and the magnetometer (heading and level); the
/// linear velocity stays at zero.
pub struct NaiveCF {
    /// Shared fusion state: device, attitude, dip, and calibration.
    pub state: FusionState,
    /// Previous gyroscope reading in FRD with its device timestamp in microseconds.
    pub prev_gyro: (Vector3<f32>, u64),
}

impl NaiveCF {
    /// Create a fusion instance over `glasses`, waiting for the first gyroscope reading.
    pub fn new(glasses: Box<dyn ARGlasses>) -> Result<Self> {
        Ok(Self::new_with_first_sample(glasses)?.0)
    }

    /// Create a fusion instance over `glasses`, waiting for the first
    /// gyroscope reading and returning it with the first AccGyro sample's
    /// device timestamp in microseconds: the shared boot loop of the
    /// estimators embedding this one, whose own previous-sample bookkeeping
    /// (e.g. the 6-DoF estimator's `prev_acc`) starts from that timestamp.
    pub(in crate::fusion) fn new_with_first_sample(
        glasses: Box<dyn ARGlasses>,
    ) -> Result<(Self, u64)> {
        let mut fusion = Self {
            state: FusionState::new(glasses),
            prev_gyro: (Vector3::zeros(), 0),
        };

        loop {
            //wait for the first non-zero gyro reading
            let next_event = fusion.state.next_event();
            if let GlassesEvent::AccGyro {
                gyroscope,
                timestamp,
                ..
            } = next_event
            {
                //if gyroscope != Vector3::zeros() {
                fusion.prev_gyro = (rub_to_frd(&gyroscope), timestamp);
                return Ok((fusion, timestamp));
                //}
            }
        }
    }

    /// Create a 3-DoF fusion instance over whichever glasses are connected, or the SimMotion device.
    pub fn any() -> Result<Box<dyn Fusion>> {
        let glasses = any_glasses_or_dummy()?;
        Ok(Box::new(Self::new(glasses)?))
    }

    /// Dead-reckoning prediction step of the complementary filter: applies the
    /// gyroscope increment $d\tilde{S}_1\, dt_1$ to the attitude before any
    /// state-sensor correction. `gyro_rub` is a RUB angular rate in rad/s, `t`
    /// a device timestamp in microseconds, and the applied increment magnitude
    /// is recorded into the gyro consistency source as the sensor-health
    /// signal. Zero elapsed time (duplicate timestamps) skips propagation
    /// and recording while still refreshing `prev_gyro`.
    pub(in crate::fusion) fn integrate_gyro(&mut self, gyro_rub: &Vector3<f32>, t: u64) {
        let gyro = rub_to_frd(gyro_rub);

        let d_t1 /*$dt_1$*/ = t - self.prev_gyro.1;
        let d_t1_f = d_t1 as f32 / naive_cf_core::GYRO_SPEED_IN_TIMESTAMP_FACTOR;
        let d_s1_t1 /*$d\tilde{S}_1\, dt_1$*/ = d_t1_f * gyro;

        if d_t1_f > 0.0 {
            let increment = UnitQuaternion::from_euler_angles(d_s1_t1.x, d_s1_t1.y, d_s1_t1.z);

            self.state.attitude *= increment;
            let _ = self
                .state
                .consistency
                .sources
                .gyro
                .record(increment.angle());
        }

        self.prev_gyro = (gyro, t);
    }

    /// Consume one AccGyro sample: the dead-reckoning prediction, then the
    /// gravity-direction state-sensor correction, then renormalization.
    /// Exposed so the estimators embedding this one (the 6-DoF estimator)
    /// run the identical attitude channel on the identical events.
    pub(in crate::fusion) fn update_acc_gyro(
        &mut self,
        accelerometer: &Vector3<f32>,
        gyroscope: &Vector3<f32>,
        timestamp: u64,
    ) {
        // dead-reckoning prediction, then the state-sensor correction
        self.integrate_gyro(gyroscope, timestamp);
        self.state.integrate_acc(accelerometer, timestamp);
        self.state.renormalize();
    }

    /// Consume one Magnetometer sample: the magnetometer attitude
    /// correction, then renormalization.
    pub(in crate::fusion) fn update_mag(&mut self, magnetometer: &Vector3<f32>, timestamp: u64) {
        self.state.integrate_mag(magnetometer, timestamp);
        self.state.renormalize();
    }

    const REGRESS_ROLL_FACTOR: f32 = 0.0;
    #[allow(dead_code)]
    pub(super) fn integrate_regress_roll(&mut self) {
        // Unused, this is only used for testing
        if !Self::REGRESS_ROLL_FACTOR.is_finite() || (Self::REGRESS_ROLL_FACTOR <= 0.0) {
            return;
        }

        let no_roll_factor = Self::REGRESS_ROLL_FACTOR.clamp(0.0, 1.0);
        let (roll, pitch, yaw) = self.state.attitude.euler_angles();
        let corrected_roll = roll * (1.0 - no_roll_factor);
        self.state.attitude = UnitQuaternion::from_euler_angles(corrected_roll, pitch, yaw);
    }
}

//unsafe impl Sync for NaiveCF {}

impl Fusion for NaiveCF {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        &mut self.state.glasses
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        self.state.attitude
    }

    fn linear_velocity(&self) -> Vector3<f32> {
        // the 3-DoF estimator never updates it; the override keeps the
        // trait output in sync with the pub state field
        self.state.linear_velocity
    }

    fn consistency(&self) -> Consistency {
        self.state.consistency
    }

    fn update(&mut self) {
        let event = self.state.next_event();
        match event {
            GlassesEvent::AccGyro {
                accelerometer,
                gyroscope,
                timestamp,
            } => {
                self.update_acc_gyro(&accelerometer, &gyroscope, timestamp);
            }

            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                // The ACC-derived gravity hint keeps the
                // calibrator's preconditioned gravity surrogate anchored to the exact
                // magnetic dip, whose learned projection
                // $\kappa / (\gamma r)$ the result reports as `dip_sin`.
                // The field's vertical component refines the fusion
                // state's own `mag_dip_sin` estimate, and the attitude
                // update compares the full field against the dipped
                // reference built from it, so the magnetometer corrects
                // the heading and, jointly with the acc filter, the
                // level.
                self.update_mag(&magnetometer, timestamp);
            }
            _ => {
                // TODO: handle KeyPress signal
            }
        }
    }
}
