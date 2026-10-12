//!
//! NaiveCF is a simple sensor fusion algorithm that uses a complementary filter.
//!
//! Complementary filter with a very simple update algorithm. With $S$ and $S^{-}$ the current and
//! previous estimated state, $d\tilde{S}_1$ a rate sensor reading (e.g. gyroscope: high frequency, high
//! drift, dead reckoning), $\tilde{S}_2$ a state sensor reading (e.g. gravity/accelerometer, magnetometer:
//! low frequency, high noise, low drift), $dt_1$ the time elapsed since the last rate sensor sample, and
//! $\mathrm{ratio}$ the blend ratio:
//!
//! $$S = \mathrm{ratio}\, (S^{-} + d\tilde{S}_1\, dt_1) + (1 - \mathrm{ratio})\, \tilde{S}_2$$
//!
//! This implies:
//!
//! - The algorithm natively supports state sensor(s) with different sampling frequency, incomplete
//!   reading, or unreliable reading, as the $\tilde{S}_2$ variable is merely an optional correction.
//! - The interpolation between $\tilde{S}_2$ and the rate-integrated prediction doesn't need to be linear
//!   or additive, e.g. 3D angular interpolation is multiplicative.
//! - The ratio can be adjusted based on quality and frequency of the state sensor(s).
//!
//! Most glasses have acc & grav/acc readings in 1 bundle, but I prefer not using this assumption and still
//! update them independently.
//!
//! The attitude pipeline shared with the other naive complementary-filter
//! estimators (gyroscope dead-reckoning, magnetometer heading correction,
//! the gravity-direction correction helper, and the device event loop)
//! lives in [`crate::fusion::_3dof::naive_cf_core`]. This estimator keeps its
//! 3-DoF-specific gravity correction, which reads the accelerometer as a
//! pure gravity-direction sensor.
//!
//! # example:
//!  TODO: fill

use nalgebra::{UnitQuaternion, Vector3};

use super::naive_cf_core;
use crate::fusion::{rub_to_frd, Consistency, Fusion, FusionState};
use crate::{any_glasses_or_dummy, ARGlasses, Error, GlassesEvent};

type Result<T> = std::result::Result<T, Error>;

pub struct NaiveCF {
    pub state: FusionState,
    pub prev_gyro: (Vector3<f32>, u64), //FRD
}

impl NaiveCF {
    pub fn new(glasses: Box<dyn ARGlasses>) -> Result<Self> {
        let mut fusion = NaiveCF {
            state: FusionState::new(glasses),
            prev_gyro: (Vector3::zeros(), 0),
        };

        loop {
            //wait for the first non-zero gyro reading
            let next_event = naive_cf_core::next_event(&mut fusion.state);
            match next_event {
                GlassesEvent::AccGyro {
                    accelerometer: _,
                    gyroscope,
                    timestamp,
                } => {
                    //if gyroscope != Vector3::zeros() {
                    fusion.prev_gyro = (rub_to_frd(&gyroscope), timestamp);
                    return Ok(fusion);
                    //}
                }
                _ => {}
            }
        }
    }

    /// Dead-reckoning prediction step; see [`naive_cf_core::integrate_gyro`].
    pub(super) fn integrate_gyro(&mut self, gyro_rub: &Vector3<f32>, t: u64) -> () {
        naive_cf_core::integrate_gyro(&mut self.state, &mut self.prev_gyro, gyro_rub, t);
    }

    /// Gravity-direction state-sensor correction; see
    /// [`naive_cf_core::integrate_acc`].
    fn integrate_acc(&mut self, acc_rub: &Vector3<f32>, t: u64) -> () {
        naive_cf_core::integrate_acc(&mut self.state, acc_rub, t);
    }

    const REGRESS_ROLL_FACTOR: f32 = 0.0;
    #[allow(dead_code)]
    pub(super) fn integrate_regress_roll(&mut self) -> () {
        // Unused, this is only used for testing
        if !Self::REGRESS_ROLL_FACTOR.is_finite() || (Self::REGRESS_ROLL_FACTOR <= 0.0) {
            return;
        }

        let no_roll_factor = Self::REGRESS_ROLL_FACTOR.clamp(0.0, 1.0);
        let (roll, pitch, yaw) = self.state.attitude.euler_angles();
        let corrected_roll = roll * (1.0 - no_roll_factor);
        self.state.attitude = UnitQuaternion::from_euler_angles(corrected_roll, pitch, yaw);
    }

    /// Magnetometer heading correction; see [`naive_cf_core::integrate_mag`].
    pub(super) fn integrate_mag(
        &mut self,
        mag_rub: &Vector3<f32>,
        calibration_use_gravity: bool,
        t: u64,
    ) -> () {
        naive_cf_core::integrate_mag(&mut self.state, mag_rub, calibration_use_gravity, t);
    }

    fn renormalize(&mut self) {
        naive_cf_core::renormalize(&mut self.state);
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

    fn update(&mut self) -> () {
        let event = naive_cf_core::next_event(&mut self.state);
        match event {
            GlassesEvent::AccGyro {
                accelerometer,
                gyroscope,
                timestamp,
            } => {
                // dead-reckoning prediction, then the state-sensor correction
                self.integrate_gyro(&gyroscope, timestamp);
                self.integrate_acc(&accelerometer, timestamp);
                self.renormalize();
            }

            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                // The ACC-derived gravity hint keeps the
                // calibrator's preconditioned gravity surrogate anchored to the exact
                // magnetic dip, whose learned projection
                // $\kappa / (\gamma r)$ the result reports as `dip_sin`.
                // The heading update uses only the field's horizontal
                // component, so the magnetometer never corrects the level
                // estimate the acc filter maintains.
                self.integrate_mag(&magnetometer, true, timestamp);
                self.renormalize();
            }
            _ => {
                // TODO: handle KeyPress signal
            }
        }
    }
}

impl dyn Fusion {
    /// Create a 3-DoF fusion instance over whichever glasses are connected, or the SimMotion device.
    pub fn any_cf_3dof() -> crate::Result<Box<dyn Fusion>> {
        let glasses = any_glasses_or_dummy()?;
        Ok(Box::new(NaiveCF::new(glasses)?))
    }
}
