use nalgebra::{Vector3, Vector4};

use crate::connection::Connection;
use crate::fusion::Fusion;

/// Start the singleton connection and its background fusion thread. Always returns 1.
#[no_mangle]
pub extern "C" fn StartConnection() -> i32 {
    Connection::start().unwrap();
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
        EULER.as_ptr()
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
        QUATERNION.as_ptr()
    }
}
