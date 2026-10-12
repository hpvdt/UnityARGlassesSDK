mod naive_cf;
pub(crate) use naive_cf::NaiveCF;

#[cfg(test)]
mod naive_cf_test;

use std::f32::consts::PI;

use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};

use crate::fusion::{Consistency, ConsistencyStatus, FusionConsistency};
use crate::{any_glasses_or_dummy, ARGlasses};

/// Sensor fusion algorithm interface: consumes raw glasses events and produces an attitude estimate.
pub trait Fusion: Send + FusionConsistency {
    /// Underlying glasses device this fusion reads events from.
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses>;
    // TODO: only declared mutable as many API of ARGlasses are also mutable

    /// primary estimation output
    /// can be used to convert to Euler angles of different conventions
    fn attitude_quaternion(&self) -> UnitQuaternion<f32>;

    /// Estimated linear velocity in the world FRD frame, in m/s.
    ///
    /// Returns zero for estimators that do not model translation (the
    /// default); the `_6dof` estimator overrides this with its tracked
    /// [`crate::fusion::FusionState::linear_velocity`].
    fn linear_velocity(&self) -> Vector3<f32> {
        Vector3::zeros()
    }

    /// Per-sensor innovation consistency tracked by the fusion algorithm.
    fn consistency(&self) -> Consistency;

    /// Consume the next sensor event and advance the attitude estimate.
    fn update(&mut self) -> ();
}

impl dyn Fusion {
    /// Create a fusion instance over whichever glasses are connected, or the SimMotion device.
    pub fn any_cf_3dof() -> crate::Result<Box<dyn Fusion>> {
        // let glasses = any_glasses()?;
        let glasses = any_glasses_or_dummy()?;
        Ok(Box::new(NaiveCF::new(glasses)?))
    }
}

/// Wraps a [`Fusion`] to output attitudes in a chosen reference frame and bias convention.
pub struct AhrsCorrection {
    fusion: Box<dyn Fusion>,
    neutral_bias: (UnitQuaternion<f32>, UnitQuaternion<f32>), // multiplicative
    euler_bias: Vector3<f32>,
    euler_order: Vector3<i16>, // each element represents an index + sign of an Euler angle axis
    quaternion_bias: (UnitQuaternion<f32>, UnitQuaternion<f32>),
    quaternion_order: Vector4<i16>,
}

impl AhrsCorrection {
    // default reference system used by most robotics applications
    //
    // Euler-Angle: forward-right-down, right hand axes, right hand rotation
    //
    // Quaternion: right hand, ijkw
    /// Correction with the standard forward-right-down (FRD) reference frame.
    pub fn frd(fusion: Box<dyn Fusion>) -> AhrsCorrection {
        AhrsCorrection {
            // defaults to RUF reference frame of Unity game engine
            fusion,
            neutral_bias: (AhrsCorrection::q_id(), AhrsCorrection::q_id()),
            euler_bias: Vector3::new(0.0, 0.0, 0.0),
            euler_order: Vector3::new(1, 2, 3),
            quaternion_bias: (AhrsCorrection::q_id(), AhrsCorrection::q_id()),
            quaternion_order: Vector4::new(1, 2, 3, 4),
        }
    }

    // reference system used by AirAPI_Windows
    // neutral heading is 90 degree pitch down
    //
    // Euler-Angle: forward-right-up, left hand axes, right hand rotation
    // CAUTION: this is a heterochiral system (using different hands for axes and rotation)
    //   in practice it should be avoided due to being highly corruptive
    //   but here it is used for backward compatibility
    //
    // Quaternion: left hand, ijkw ?
    /// Correction matching the AirAPI_Windows reference frame, with neutral heading 90° pitch down.
    pub fn left_fru_down(fusion: Box<dyn Fusion>) -> AhrsCorrection {
        AhrsCorrection {
            // defaults to RUF reference frame of Unity game engine
            fusion,
            // neutral_bias: UnitQuaternion::from_euler_angles(0.0, PI * 0.5, 0.0),
            neutral_bias: (AhrsCorrection::q_id(), AhrsCorrection::q_id()),
            euler_bias: Vector3::new(0.0, PI * 0.5, 0.0),
            euler_order: Vector3::new(1, 2, -3),
            quaternion_bias: (
                // UnitQuaternion::from_euler_angles(0.0, PI * 0.5, 0.0),
                AhrsCorrection::q_id(),
                UnitQuaternion::from_euler_angles(0.0, PI * 0.5, 0.0),
                // AHRS::q_id(),
            ),
            quaternion_order: Vector4::new(2, 1, -3, 4),
        }
    }

    fn q_id() -> UnitQuaternion<f32> {
        UnitQuaternion::identity()
    }

    fn attitude_quaternion_frd(&self) -> UnitQuaternion<f32> {
        let original = self.fusion.attitude_quaternion();
        let corrected = self.neutral_bias.0 * original * self.neutral_bias.1;
        corrected
    }

    /// Attitude as Euler angles (roll, pitch, yaw) in the configured frame, in radians.
    pub fn attitude_euler_rad(&self) -> Vector3<f32> {
        let (roll, pitch, yaw) = self.attitude_quaternion_frd().euler_angles();
        // TODO: add roundtrip test
        let frd = Vector3::new(roll, pitch, yaw);
        let biased = frd + self.euler_bias;
        let ordered = self.euler_order.map(|v| {
            let index = (v.abs() - 1) as usize;
            let vv = biased.get(index).unwrap();
            let vv_with_sign = vv * (v.signum() as f32);
            vv_with_sign
        });
        ordered
    }

    /// Attitude as Euler angles (roll, pitch, yaw) in the configured frame, in degrees.
    pub fn attitude_euler_deg(&self) -> Vector3<f32> {
        self.attitude_euler_rad().map(|x| x.to_degrees())
    }

    /// Returns the non-overridable fusion consistency verdict.
    pub fn consistency_status(&self) -> ConsistencyStatus {
        <Self as FusionConsistency>::consistency_status(self)
    }
}

impl Fusion for AhrsCorrection {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        self.fusion.glasses()
    }

    fn linear_velocity(&self) -> Vector3<f32> {
        // world-FRD state; the wrapper's attitude conventions do not apply
        self.fusion.linear_velocity()
    }

    fn consistency(&self) -> Consistency {
        self.fusion.consistency()
    }

    fn update(&mut self) -> () {
        self.fusion.update()
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        let q_raw = self.attitude_quaternion_frd();

        let q_biased = self.quaternion_bias.0 * q_raw * self.quaternion_bias.1;

        // let q_biased = q_raw * self.quaternion_bias;
        let biased = q_biased.coords;

        let ordered = self.quaternion_order.map(|v| {
            let index = (v.abs() - 1) as usize;
            let vv = biased.get(index).unwrap();
            let vv_with_sign = vv * (v.signum() as f32);
            vv_with_sign
        });

        let q_ordered = UnitQuaternion::from_quaternion(Quaternion::new(
            ordered[3], ordered[0], ordered[1], ordered[2],
        ));

        q_ordered
    }
}
