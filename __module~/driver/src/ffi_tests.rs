//! Tests of the C ABI surface: the linear-velocity export's presentation and
//! its singleton-connection plumbing.

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Ordering shared by the poison-flag store in [`PanickingFusion::update`]
/// and its polls in [`wait_poisoned`].
const POISONED_ORDERING: Ordering = Ordering::SeqCst;

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

/// Estimator fixture whose first update panics, poisoning the fusion state
/// and killing the background thread, so exports report the failure through
/// their `error` arguments instead of panicking across the C ABI.
struct PanickingFusion {
    glasses: Box<dyn ARGlasses>,
    /// Set immediately before the update panics, so tests can wait for the
    /// poison instead of sleeping for a fixed time.
    poisoned: Arc<AtomicBool>,
}

impl Fusion for PanickingFusion {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        &mut self.glasses
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::identity()
    }

    fn linear_velocity(&self) -> Vector3<f32> {
        Vector3::zeros()
    }

    fn consistency(&self) -> Consistency {
        Consistency::attitude_defaults()
    }

    fn update(&mut self) {
        self.poisoned.store(true, POISONED_ORDERING);
        panic!("poison the fusion state to kill the connection thread");
    }
}

/// Wait until `poisoned` is set, which happens while the fusion mutex is
/// held by the panicking background thread.
fn wait_poisoned(poisoned: &Arc<AtomicBool>) {
    while !poisoned.load(POISONED_ORDERING) {
        thread::sleep(Duration::from_millis(1));
    }
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

    let mut error: *const c_char = std::ptr::null();
    let ptr = GetLinearVelocity(&mut error);
    assert!(error.is_null(), "unexpected staged error: {error:?}");

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

    let mut error: *const c_char = std::ptr::null();
    let euler_ptr = GetEuler(&mut error);
    let quaternion_ptr = GetQuaternion(&mut error);
    let velocity_ptr = GetLinearVelocity(&mut error);
    assert!(error.is_null(), "unexpected staged error: {error:?}");

    assert_eq!(GetEuler(&mut error), euler_ptr);
    assert_eq!(GetQuaternion(&mut error), quaternion_ptr);
    assert_eq!(GetLinearVelocity(&mut error), velocity_ptr);

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

    let mut error: *const c_char = std::ptr::null();

    // SAFETY: the pointers are the exports' static storage, read here
    // before any further calls overwrite them
    let euler = unsafe { std::slice::from_raw_parts(GetEuler(&mut error), 3) };
    let wxyz = unsafe { std::slice::from_raw_parts(GetQuaternion(&mut error), 4) };
    assert!(error.is_null(), "unexpected staged error: {error:?}");

    assert_eq!(euler, expected_euler.as_slice());
    assert_eq!(wxyz, expected_wxyz.as_slice());

    Connection::stop().unwrap();
}

/// The connection lifecycle exports report success codes: starting builds
/// the 6-DoF fusion over the connected glasses (or the sim fallback), every
/// staged reading stays finite, and stopping reports success both while a
/// connection is running and when none exists. The `error` out slot stays
/// null on every success, and a null slot is accepted to opt out.
#[test]
#[serial]
fn connection_lifecycle_exports_report_success() {
    Connection::stop().unwrap();

    let mut error: *const c_char = std::ptr::null();

    assert_eq!(StartConnection(&mut error), 1);
    assert!(error.is_null(), "unexpected staged error: {error:?}");

    // SAFETY: the pointer is the export's static storage, read here before
    // any further call
    let euler = unsafe { std::slice::from_raw_parts(GetEuler(&mut error), 3) };
    assert!(euler.iter().all(|v| v.is_finite()));
    assert!(error.is_null(), "unexpected staged error: {error:?}");

    // a null `error` argument opts out of the failure report
    assert_eq!(StartConnection(std::ptr::null_mut()), 1);

    assert_eq!(StopConnection(&mut error), 1);
    assert!(error.is_null(), "unexpected staged error: {error:?}");
    assert_eq!(StopConnection(&mut error), 1);
    assert!(error.is_null(), "unexpected staged error: {error:?}");
}

/// A failed export reports its failure through the `error` out argument: a
/// non-null pointer to a NUL-terminated message, valid until the next
/// module call, naming the failed operation and its cause. The connection
/// thread's panic is the failure exercised here, because it is available on
/// every host, unlike a missing device.
#[test]
#[serial]
fn failed_export_stages_the_error_message() {
    Connection::stop().unwrap();

    let poisoned = Arc::new(AtomicBool::new(false));
    Connection::start(Some(Box::new(PanickingFusion {
        glasses: Box::new(Dummy {}),
        poisoned: poisoned.clone(),
    })))
    .unwrap();

    wait_poisoned(&poisoned);

    let mut error: *const c_char = std::ptr::null();
    assert_eq!(StopConnection(&mut error), 0);
    assert!(!error.is_null());

    // SAFETY: the message is staged storage, valid until the next module
    // call and read here before any further one
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    assert!(
        message.contains("connection thread panicked"),
        "unexpected staged message: {message}"
    );
}

/// While the fusion state is poisoned, the `Get*` exports keep the last
/// staged reading and report the failure through `error`, and a null
/// `error` argument opts out of the report.
#[test]
#[serial]
fn getters_report_a_poisoned_fusion_state() {
    Connection::stop().unwrap();

    let poisoned = Arc::new(AtomicBool::new(false));
    Connection::start(Some(Box::new(PanickingFusion {
        glasses: Box::new(Dummy {}),
        poisoned: poisoned.clone(),
    })))
    .unwrap();

    wait_poisoned(&poisoned);

    let mut error: *const c_char = std::ptr::null();
    let euler_ptr = GetEuler(&mut error);
    assert!(!error.is_null());

    // SAFETY: the message is staged storage, valid until the next module
    // call and read here before any further one
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    assert!(
        message.contains("fusion state poisoned"),
        "unexpected staged message: {message}"
    );

    // the last staged reading is kept, and the null `error` argument opts out
    assert_eq!(GetEuler(std::ptr::null_mut()), euler_ptr);

    // the poisoned thread's exit surfaces as the join failure
    Connection::stop().unwrap_err();
}
