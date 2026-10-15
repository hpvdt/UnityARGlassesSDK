//! Tests of the C ABI surface: the linear-velocity export's presentation and
//! its singleton-connection plumbing.

use nalgebra::{UnitQuaternion, Vector3, Vector4};

use serial_test::serial;

use crate::connection::Connection;
use crate::ffi::{GetEuler, GetLinearVelocity, GetQuaternion, StartConnection, StopConnection};
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

/// Every `Get*` export stages its reading in a fixed buffer, so repeated
/// calls return the same address and the C caller's "valid until the next
/// call" copy-out window holds.
#[test]
#[serial]
fn getters_stage_into_stable_pointers() {
    Connection::stop().unwrap();

    let fusion = FixedVelocityFusion {
        glasses: Box::new(Dummy {}),
        velocity: Vector3::new(0.3, -0.2, 0.1),
    };
    Connection::start(Some(Box::new(fusion))).unwrap();

    let euler_ptr = GetEuler();
    let quaternion_ptr = GetQuaternion();
    let velocity_ptr = GetLinearVelocity();

    assert_eq!(GetEuler(), euler_ptr);
    assert_eq!(GetQuaternion(), quaternion_ptr);
    assert_eq!(GetLinearVelocity(), velocity_ptr);

    Connection::stop().unwrap();
}

/// `GetEuler` and `GetQuaternion` mirror the connection's reading verbatim:
/// the staged Euler components equal `attitude_euler_deg`, and the staged
/// quaternion is the same estimate re-ordered from IJKW to WXYZ.
#[test]
#[serial]
fn getters_mirror_the_connection_reading() {
    Connection::stop().unwrap();

    let fusion = FixedVelocityFusion {
        glasses: Box::new(Dummy {}),
        velocity: Vector3::new(0.3, -0.2, 0.1),
    };
    Connection::start(Some(Box::new(fusion))).unwrap();

    let expected_euler = Connection::read_fusion(&|ff| ff.attitude_euler_deg()).unwrap();
    let expected_wxyz = Connection::read_fusion(&|ff| {
        let quaternion = ff.attitude_quaternion();
        let ijkw = quaternion.as_vector();
        Vector4::new(ijkw[3], ijkw[0], ijkw[1], ijkw[2])
    })
    .unwrap();

    // SAFETY: the pointers are the exports' static storage, read here
    // before any further calls overwrite them
    let euler = unsafe { std::slice::from_raw_parts(GetEuler(), 3) };
    let wxyz = unsafe { std::slice::from_raw_parts(GetQuaternion(), 4) };

    assert_eq!(euler, expected_euler.as_slice());
    assert_eq!(wxyz, expected_wxyz.as_slice());

    Connection::stop().unwrap();
}

/// The connection lifecycle exports report success codes: starting builds
/// the 6-DoF fusion over the connected glasses (or the sim fallback), every
/// staged reading stays finite, and stopping reports success both while a
/// connection is running and when none exists.
#[test]
#[serial]
fn connection_lifecycle_exports_report_success() {
    Connection::stop().unwrap();

    assert_eq!(StartConnection(), 1);

    // SAFETY: the pointer is the export's static storage, read here before
    // any further call
    let euler = unsafe { std::slice::from_raw_parts(GetEuler(), 3) };
    assert!(euler.iter().all(|v| v.is_finite()));

    assert_eq!(StopConnection(), 1);
    assert_eq!(StopConnection(), 1);
}
