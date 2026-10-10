/*
high level interface of glasses & state estimation, with the following built-in fusion pipeline:

(the first version should only use complementary filter for simplicity and sanity test)

- roll/pitch <= acc + gyro (complementary filter)
  - assuming that acc vector always pointed up, spacecraft moving in that direction can create 1G artificial gravity
    - TODO: this obviously assumes no steadily accelerating frame, at which point up d_acc has to be used for correction
- gyro-yaw <= gyro (integrate over time)
- mag-dip <= the calibrator's learned dip projection $\kappa / (\gamma r)$, reported as `dip_sin` (state of
  the calibrator's gravity surrogate, not of the attitude estimate)
- mag-yaw <= mag horizontal component + roll/pitch (never corrects the level)
- yaw <= mag-yaw + gyro-gyro (complementary filter)
  - TODO: add EKF/ESKF (error-state/multiplicatory KF, https://arxiv.org/abs/1711.02508)

CAUTION: unlike [[GlassesEvent]], all states & outputs should use FRD reference frame
 (forward, right, down, corresponding to roll, pitch, yaw in Euler angles-represented rotation)

FRD is the standard frame for aerospace, and is also the default frame for NALgebra
*/
use std::fmt;

use nalgebra::{UnitQuaternion, Vector3};

use crate::ARGlasses;

mod _3dof;
pub use _3dof::{AhrsCorrection, Fusion};

mod naive_cf_core;

mod mag;
pub use mag::{
    BadCalibration, BadMagCause, BadReading, CalibrationQuality, MagCalibrationResult,
    MagCalibrator,
};

mod consistency;
pub use consistency::{
    Consistency, ConsistencyStatus, EmaTracking, FusionConsistency, SourceConsistency,
};

/// Converts a raw sensor vector from RUB (right, up, back) into FRD (forward, right, down).
pub fn rub_to_frd(v: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(-v.z, v.x, -v.y)
}

/// Per-sensor (acc/gyro/mag) triplet of homogeneous values.
#[derive(Clone, Copy, Debug, Default)]
pub struct NineAxis<T> {
    /// Accelerometer correction.
    pub acc: T,
    /// Gyroscope integration increment.
    pub gyro: T,
    /// Magnetometer correction.
    pub mag: T,
}

impl<T: fmt::Display> fmt::Display for NineAxis<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acc: {}, gyro: {}, mag: {}",
            self.acc, self.gyro, self.mag
        )
    }
}

/// Shared fusion algorithm state: glasses device, current attitude, and magnetometer calibration.
pub struct FusionState {
    /// Glasses device the state reads events from.
    pub glasses: Box<dyn ARGlasses>,

    // following data will be updated in memory directly,
    /// Latest attitude estimate as a unit quaternion.
    pub attitude: UnitQuaternion<f32>, /*$S$*/

    /// Linear velocity estimate in the world FRD frame, in m/s; reserved for
    /// the upcoming 6DoF fusion, not yet updated by any estimator.
    pub linear_velocity: Vector3<f32>,

    /// Per-sensor innovation consistency.
    pub consistency: Consistency,

    // mag calibration state, will be used by all Fusion impls
    // The buffer must outlast a single motion pattern: with too few samples
    // it only spans one rotation segment, the retained readings cover a
    // near-planar circle on the sphere, and the ellipsoid fit is free to
    // drift along the unobserved axis (seen as >20 deg worst-case heading
    // error in the SimMotion integration test with 255 samples).
    /// Magnetometer calibration state shared by all fusion implementations.
    ///
    /// Boxed: the calibrator is ~104 KB inline, and constructing it by value
    /// through the `FusionState::new`/`NaiveCF::new` chain overflowed the
    /// 1 MiB Windows main-thread stack in debug builds.
    pub mag_calibrator: Box<MagCalibrator<1023>>,
}

impl FusionState {
    /// Creates a shared fusion state with identity attitude and empty
    /// calibration state.
    pub fn new(glasses: Box<dyn ARGlasses>) -> Self {
        Self {
            glasses,
            attitude: UnitQuaternion::identity(),
            linear_velocity: Vector3::identity(),
            consistency: Consistency::attitude_defaults(),
            mag_calibrator: Box::new(MagCalibrator::new()),
        }
    }
}
