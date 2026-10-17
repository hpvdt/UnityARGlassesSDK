/*! The C ABI of the singleton connection consumed by the Unity integration.
 *
 * Failures are reported through an out-position `error` argument that every
 * export accepts (`*mut *const c_char`): it is left null on success and
 * otherwise points at the failure message, staged module-wide. The message
 * pointer is valid only until the next call into this module, so the caller
 * must copy it out before any further call. The `error` argument itself may
 * be null to opt out of the report. Callers must not invoke the exports
 * concurrently from multiple threads.
 */

// Raw pointer arguments are this ABI's error channel: its `unsafe`
// obligations are documented per export and cannot be expressed to the C
// caller, so the pointer-dereference lint is allowed module-wide.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::UnsafeCell;
use std::ffi::{c_char, CString};

use nalgebra::{Vector3, Vector4};

use crate::connection::Connection;
use crate::fusion::_6dof::naive_cf::NaiveCF;
use crate::fusion::Fusion;

/// Interior-mutability staging cell for the buffers handed out by the `Get*`
/// C ABI getters and for the staged error message.
///
/// The value is only ever accessed through raw pointers, so no Rust reference
/// into the cell is created; the handed-out pointer crosses the C ABI and is
/// governed by the getters' documented caller contract.
struct FfiCell<T>(UnsafeCell<T>);

// SAFETY: `FfiCell` is private to this module and only reached by the `Get*`
// C ABI getters and the error-staging helper, which the C caller must not
// invoke concurrently from multiple threads; no Rust reference into the cell
// is ever shared across threads.
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

/// Error message staged for the `error` out-arguments: the message of the
/// last export call, or null when it succeeded.
static LAST_ERROR: FfiCell<Option<CString>> = FfiCell::new(None);

/// Stage the failure of the `what` operation as the module's last error and
/// point the caller's `error` out slot at its message, or clear the slot
/// with null when `failure` is none. Interior NUL bytes are stripped from
/// the message so the C string cannot be truncated early.
///
/// # Safety
///
/// The caller must uphold the module-level `error` contract: no other thread
/// may call into this module while this runs, and `error` must be null or
/// point to a caller-owned slot for the duration of the call.
unsafe fn stage_error(error: *mut *const c_char, what: &str, failure: Option<&crate::Error>) {
    let staged = failure.map(|e| {
        let sanitized: String = format!("Error {what}: {e}")
            .chars()
            .filter(|c| *c != '\0')
            .collect();
        // `sanitized` holds no NUL, so the conversion cannot fail and the
        // default only appeases the type system.
        CString::new(sanitized).unwrap_or_default()
    });
    let staged_ptr = match staged.as_ref() {
        Some(message) => message.as_ptr(),
        None => std::ptr::null(),
    };

    // SAFETY: the module-level `error` contract guarantees that no other
    // thread touches the staged error cell while it is written here.
    unsafe { LAST_ERROR.write(staged) };

    if !error.is_null() {
        // SAFETY: `error` is a caller-owned slot for the duration of the
        // call, per the module-level contract.
        unsafe { *error = staged_ptr };
    }
}

/// Start the singleton connection and its background fusion thread with the
/// default 6-DoF fusion: the same attitude estimate as the 3-DoF one plus
/// the world-frame linear velocity exposed by `GetLinearVelocity`. Returns
/// 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn StartConnection(error: *mut *const c_char) -> i32 {
    let result = NaiveCF::any().and_then(|fusion| Connection::start(Some(fusion)));
    // SAFETY: Unity calls this export from a single thread, per the
    // module-level `error` contract.
    unsafe { stage_error(error, "starting connection", result.as_ref().err()) };
    i32::from(result.is_ok())
}

/// Stop the singleton connection and join its background fusion thread.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn StopConnection(error: *mut *const c_char) -> i32 {
    let result = Connection::stop();
    // SAFETY: Unity calls this export from a single thread, per the
    // module-level `error` contract.
    unsafe { stage_error(error, "stopping connection", result.as_ref().err()) };
    i32::from(result.is_ok())
}

/// Latest Euler reading staged for [`GetEuler`].
static EULER: FfiCell<Vector3<f32>> = FfiCell::new(Vector3::new(0.0, 0.0, 0.0));

/// Latest attitude as Euler angles (roll, pitch, yaw) in degrees; pointer to 3 `f32`
/// valid until the next call. Keeps the last successfully read value (initially
/// zero) while the connection cannot be read, with `*error` then pointing at
/// the failure message.
#[no_mangle]
pub extern "C" fn GetEuler(error: *mut *const c_char) -> *const f32 {
    let read = Connection::read_fusion(&|ff| ff.attitude_euler_deg());

    // SAFETY: Unity calls this getter from a single thread, per the
    // module-level `error` contract.
    unsafe { stage_error(error, "reading euler angles", read.as_ref().err()) };

    if let Ok(euler) = read {
        // SAFETY: the C caller copies the values out before the next call,
        // as the contract above states.
        unsafe { EULER.write(euler) };
    }
    EULER.as_ptr().cast::<f32>()
}

/// Latest WXYZ quaternion staged for [`GetQuaternion`].
static QUATERNION: FfiCell<Vector4<f32>> = FfiCell::new(Vector4::new(0.0, 0.0, 0.0, 0.0));

/// Latest attitude as a WXYZ quaternion; pointer to 4 `f32` valid until the next
/// call. Keeps the last successfully read value (initially zero) while the
/// connection cannot be read, with `*error` then pointing at the failure
/// message.
#[no_mangle]
pub extern "C" fn GetQuaternion(error: *mut *const c_char) -> *const f32 {
    let read = Connection::read_fusion(&|ff| {
        let quaternion = ff.attitude_quaternion();
        let ijkw = quaternion.as_vector();
        Vector4::new(ijkw[3], ijkw[0], ijkw[1], ijkw[2])
    });

    // SAFETY: Unity calls this getter from a single thread, per the
    // module-level `error` contract.
    unsafe { stage_error(error, "reading quaternion", read.as_ref().err()) };

    if let Ok(wijk) = read {
        // SAFETY: the C caller copies the values out before the next call,
        // as the contract above states.
        unsafe { QUATERNION.write(wijk) };
    }
    QUATERNION.as_ptr().cast::<f32>()
}

/// Latest FRU linear velocity staged for [`GetLinearVelocity`].
static LINEAR_VELOCITY: FfiCell<Vector3<f32>> = FfiCell::new(Vector3::new(0.0, 0.0, 0.0));

/// Latest estimated linear velocity in the left-handed forward-right-up (FRU)
/// component order matching the attitude presentation of `GetEuler`, in
/// m/s: the components along forward, right, up, that is
/// `(v.x, v.y, -v.z)` of the world-FRD estimate. Pointer to 3 `f32` valid
/// until the next call. Keeps the last successfully read value (initially
/// zero) while the connection cannot be read, with `*error` then pointing
/// at the failure message.
#[no_mangle]
pub extern "C" fn GetLinearVelocity(error: *mut *const c_char) -> *const f32 {
    let read = Connection::read_fusion(&|ff| ff.linear_velocity());

    // SAFETY: Unity calls this getter from a single thread, per the
    // module-level `error` contract.
    unsafe { stage_error(error, "reading linear velocity", read.as_ref().err()) };

    if let Ok(velocity) = read {
        // SAFETY: the C caller copies the values out before the next call,
        // as the contract above states.
        unsafe {
            LINEAR_VELOCITY.write(Vector3::new(velocity.x, velocity.y, -velocity.z));
        }
    }
    LINEAR_VELOCITY.as_ptr().cast::<f32>()
}

#[cfg(test)]
#[path = "ffi_tests.rs"]
mod ffi_tests;
