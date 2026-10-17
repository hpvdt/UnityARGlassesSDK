use std::cell::UnsafeCell;

use nalgebra::{Vector3, Vector4};

use crate::connection::Connection;
use crate::fusion::_6dof::naive_cf::NaiveCF;
use crate::fusion::Fusion;

// FIXME: too many "println!" that will never be read, they should be removed
//  to tell C# program if the operation is successful, each FFI function should accept a mutable "error" argument in out position, which will be set to the error message if error occur
//  this revision should affect both rust & C# code

/// Start the singleton connection and its background fusion thread with the
/// default 6-DoF fusion: the same attitude estimate as the 3-DoF one plus
/// the world-frame linear velocity exposed by `GetLinearVelocity`. Returns
/// 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn StartConnection() -> i32 {
    match NaiveCF::any().and_then(|fusion| Connection::start(Some(fusion))) {
        Ok(()) => {
            // println!("connection started");
            1
        }
        Err(e) => {
            println!("Error starting connection: {}", e);
            0
        }
    }
}

/// Stop the singleton connection and join its background fusion thread.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn StopConnection() -> i32 {
    match Connection::stop() {
        Ok(()) => {
            // println!("connection stopped");
            1
        }
        Err(e) => {
            println!("Error stopping connection: {}", e);
            0
        }
    }
}

/// Interior-mutability staging cell for the buffers handed out by the `Get*`
/// C ABI getters.
///
/// The value is only ever accessed through raw pointers, so no Rust reference
/// into the cell is created; the handed-out pointer crosses the C ABI and is
/// governed by the getters' documented caller contract.
struct FfiCell<T>(UnsafeCell<T>);

// SAFETY: `FfiCell` is private to this module and only reached by the `Get*`
// getters, which the C caller must not invoke concurrently from multiple
// threads; no Rust reference into the cell is ever shared across threads.
unsafe impl<T> Sync for FfiCell<T> {}

impl<T> FfiCell<T> {
    const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Overwrite the staged value.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no other thread reads or writes the
    /// cell while this runs.
    unsafe fn write(&self, value: T) {
        *self.0.get() = value;
    }

    /// Read-only pointer to the staged value, valid until the next [`FfiCell::write`].
    fn as_ptr(&self) -> *const T {
        self.0.get()
    }
}

/// Latest Euler reading staged for [`GetEuler`].
static EULER: FfiCell<Vector3<f32>> = FfiCell::new(Vector3::new(0.0, 0.0, 0.0));

/// Latest attitude as Euler angles (roll, pitch, yaw) in degrees; pointer to 3 `f32`
/// valid until the next call. Keeps the last successfully read value (initially
/// zero) while the connection cannot be read.
#[no_mangle]
pub extern "C" fn GetEuler() -> *const f32 {
    let read = Connection::read_fusion(&|ff| ff.attitude_euler_deg());

    // SAFETY: Unity calls this getter from a single thread; the C caller
    // copies the values out before the next call, as the contract above states.
    match read {
        Ok(euler) => unsafe { EULER.write(euler) },
        Err(e) => println!("Error reading euler angles: {}", e),
    }
    EULER.as_ptr().cast::<f32>()
}

/// Latest WXYZ quaternion staged for [`GetQuaternion`].
static QUATERNION: FfiCell<Vector4<f32>> = FfiCell::new(Vector4::new(0.0, 0.0, 0.0, 0.0));

/// Latest attitude as a WXYZ quaternion; pointer to 4 `f32` valid until the next
/// call. Keeps the last successfully read value (initially zero) while the
/// connection cannot be read.
#[no_mangle]
pub extern "C" fn GetQuaternion() -> *const f32 {
    let read = Connection::read_fusion(&|ff| {
        let quaternion = ff.attitude_quaternion();
        let ijkw = quaternion.as_vector();
        Vector4::new(ijkw[3], ijkw[0], ijkw[1], ijkw[2])
    });

    // SAFETY: Unity calls this getter from a single thread; the C caller
    // copies the values out before the next call, as the contract above states.
    match read {
        Ok(wijk) => unsafe { QUATERNION.write(wijk) },
        Err(e) => println!("Error reading quaternion: {}", e),
    }
    QUATERNION.as_ptr().cast::<f32>()
}

/// Latest FRU linear velocity staged for [`GetLinearVelocity`].
static LINEAR_VELOCITY: FfiCell<Vector3<f32>> = FfiCell::new(Vector3::new(0.0, 0.0, 0.0));

/// Latest estimated linear velocity in the left-handed forward-right-up (FRU)
/// component order matching the attitude presentation of `GetEuler`, in
/// m/s: the components along forward, right, up, that is
/// `(v.x, v.y, -v.z)` of the world-FRD estimate. Pointer to 3 `f32` valid
/// until the next call.
#[no_mangle]
pub extern "C" fn GetLinearVelocity() -> *const f32 {
    let read = Connection::read_fusion(&|ff| ff.linear_velocity());

    // SAFETY: the cell is only written here, under the connection's
    // reader pause shared by GetEuler/GetQuaternion, and handed out as a
    // pointer valid until the next call; callers must not invoke the Get*
    // family concurrently from multiple threads.
    match read {
        Ok(velocity) => unsafe {
            LINEAR_VELOCITY.write(Vector3::new(velocity.x, velocity.y, -velocity.z));
        },
        Err(e) => println!("Error reading linear velocity: {}", e),
    }
    LINEAR_VELOCITY.as_ptr().cast::<f32>()
}

#[cfg(test)]
#[path = "ffi_tests.rs"]
mod ffi_tests;
