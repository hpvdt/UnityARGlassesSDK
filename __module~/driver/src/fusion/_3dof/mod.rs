pub(super) mod naive_cf;

pub(in crate::fusion) mod naive_cf_core;
#[cfg(test)]
mod naive_cf_test;

use std::f32::consts::PI;

use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};

use crate::fusion::{Consistency, ConsistencyStatus, Fusion, FusionConsistency};
use crate::ARGlasses;

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
    pub fn frd(fusion: Box<dyn Fusion>) -> Self {
        Self {
            // defaults to RUF reference frame of Unity game engine
            fusion,
            neutral_bias: (Self::q_id(), Self::q_id()),
            euler_bias: Vector3::new(0.0, 0.0, 0.0),
            euler_order: Vector3::new(1, 2, 3),
            quaternion_bias: (Self::q_id(), Self::q_id()),
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
    pub fn left_fru_down(fusion: Box<dyn Fusion>) -> Self {
        Self {
            // defaults to RUF reference frame of Unity game engine
            fusion,
            // neutral_bias: UnitQuaternion::from_euler_angles(0.0, PI * 0.5, 0.0),
            neutral_bias: (Self::q_id(), Self::q_id()),
            euler_bias: Vector3::new(0.0, PI * 0.5, 0.0),
            euler_order: Vector3::new(1, 2, -3),
            quaternion_bias: (
                // UnitQuaternion::from_euler_angles(0.0, PI * 0.5, 0.0),
                Self::q_id(),
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
        self.neutral_bias.0 * original * self.neutral_bias.1
    }

    /// Attitude as Euler angles (roll, pitch, yaw) in the configured frame, in radians.
    pub fn attitude_euler_rad(&self) -> Vector3<f32> {
        let (roll, pitch, yaw) = self.attitude_quaternion_frd().euler_angles();
        // TODO: add roundtrip test
        let frd = Vector3::new(roll, pitch, yaw);
        let biased = frd + self.euler_bias;
        self.euler_order.map(|v| {
            let index = (v.abs() - 1) as usize;
            let vv = biased.get(index).unwrap();
            vv * (v.signum() as f32)
        })
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

    fn update(&mut self) {
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
            vv * (v.signum() as f32)
        });

        UnitQuaternion::from_quaternion(Quaternion::new(
            ordered[3], ordered[0], ordered[1], ordered[2],
        ))
    }
}
