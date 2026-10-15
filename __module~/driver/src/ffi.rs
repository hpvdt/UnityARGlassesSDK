use nalgebra::{Vector3, Vector4};

use crate::connection::Connection;
use crate::fusion::_6dof::naive_cf::NaiveCF;
use crate::fusion::Fusion;

/// Start the singleton connection and its background fusion thread with the
/// 6-DoF fusion: the same attitude estimate as the default 3-DoF one plus
/// the world-frame linear velocity exposed by `GetLinearVelocity`. Always
/// returns 1.
///
/// A `Get*` reader called before this lazily creates the singleton with the
/// default 3-DoF fusion, whose velocity estimate is always zero, and the
/// fusion requested here is then ignored; start the connection before
/// reading.
#[no_mangle]
pub extern "C" fn StartConnection() -> i32 {
    Connection::start(Some(NaiveCF::any().unwrap())).unwrap();
    // println!("connection started");
    1
    // .map_or_else(|_| 1, |_| 0)
}

/// Stop the singleton connection and join its background fusion thread. Always returns 1.
#[no_mangle]
pub extern "C" fn StopConnection() -> i32 {
    Connection::stop().unwrap();
    // println!("connection stopped");
    1
    // .map_or_else(|_| 1, |_| 0)
}

static mut EULER: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);

/// Latest attitude as Euler angles (roll, pitch, yaw) in degrees; pointer to 3 `f32` valid until the next call.
#[no_mangle]
pub extern "C" fn GetEuler() -> *const f32 {
    unsafe {
        let euler = Connection::read_fusion(&|ff| ff.attitude_euler_deg()).unwrap();

        EULER = euler;
        std::ptr::addr_of!(EULER).cast::<f32>()
    }
}

static mut QUATERNION: Vector4<f32> = Vector4::new(0.0, 0.0, 0.0, 0.0);

/// Latest attitude as a WXYZ quaternion; pointer to 4 `f32` valid until the next call.
#[no_mangle]
pub extern "C" fn GetQuaternion() -> *const f32 {
    unsafe {
        let quaternion = Connection::read_fusion(&|ff| ff.attitude_quaternion()).unwrap();

        let ijkw = quaternion.as_vector();
        let wijk = Vector4::new(ijkw[3], ijkw[0], ijkw[1], ijkw[2]);

        QUATERNION = wijk;
        std::ptr::addr_of!(QUATERNION).cast::<f32>()
    }
}

static mut LINEAR_VELOCITY: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);

/// Latest estimated linear velocity in the left-handed forward-right-up (FRU)
/// component order matching the attitude presentation of `GetEuler`, in
/// m/s: the components along forward, right, up, that is
/// `(v.x, v.y, -v.z)` of the world-FRD estimate. Pointer to 3 `f32` valid
/// until the next call; keeps the last successfully read value (initially
/// zero) while the connection cannot be read.
#[no_mangle]
pub extern "C" fn GetLinearVelocity() -> *const f32 {
    let read = Connection::read_fusion(&|ff| ff.linear_velocity());

    // SAFETY: the static is only written here, under the connection's
    // reader pause shared by GetEuler/GetQuaternion, and handed out as a
    // pointer valid until the next call; callers must not invoke the Get*
    // family concurrently from multiple threads.
    unsafe {
        match read {
            Ok(velocity) => LINEAR_VELOCITY = Vector3::new(velocity.x, velocity.y, -velocity.z),
            Err(e) => println!("Error reading linear velocity: {}", e),
        }
        std::ptr::addr_of!(LINEAR_VELOCITY).cast::<f32>()
    }
}

#[cfg(test)]
#[path = "ffi_tests.rs"]
mod ffi_tests;
