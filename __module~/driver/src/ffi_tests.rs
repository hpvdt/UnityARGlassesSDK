//! Tests of the C ABI surface: the linear-velocity export's presentation and
//! its singleton-connection plumbing.

use nalgebra::{UnitQuaternion, Vector3};

use serial_test::serial;

use crate::connection::Connection;
use crate::ffi::GetLinearVelocity;
use crate::fusion::{Consistency, Fusion};
use crate::sim::Dummy;
use crate::ARGlasses;

/// Estimator fixture returning a fixed attitude and linear velocity without
/// consuming sensor events, so the export's presentation is asserted
/// deterministically.
struct FixedVelocityFusion {
    glasses: Box<dyn ARGlasses>,
    velocity: Vector3<f32>,
}

impl Fusion for FixedVelocityFusion {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        &mut self.glasses
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::identity()
    }

    fn linear_velocity(&self) -> Vector3<f32> {
        self.velocity
    }

    fn consistency(&self) -> Consistency {
        Consistency::attitude_defaults()
    }

    fn update(&mut self) {}
}

/// The export reads the connection's `Fusion::linear_velocity` through the
/// `AhrsCorrection` wrapper and mirrors it into the left-handed
/// forward-right-up presentation shared with `GetEuler`: the down component
/// negated into up, so `(F, R, U) = (v.x, v.y, -v.z)`.
#[test]
#[serial]
fn get_linear_velocity_presents_fru_order() {
    Connection::stop().unwrap();

    let fusion = FixedVelocityFusion {
        glasses: Box::new(Dummy {}),
        velocity: Vector3::new(0.3, -0.2, 0.1),
    };
    Connection::start(Some(Box::new(fusion))).unwrap();

    let ptr = GetLinearVelocity();
    // SAFETY: the pointer is the export's static storage, valid until the
    // next call and read here before any further one
    let fru = unsafe { std::slice::from_raw_parts(ptr, 3) };

    assert_eq!(fru, [0.3, -0.2, -0.1].as_slice());

    Connection::stop().unwrap();
}
