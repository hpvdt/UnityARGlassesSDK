// Copyright (C) 2023, Alex Badics
// This file is part of ar-drivers-rs
// Licensed under the MIT license. See LICENSE file in the project root for details.

use std::ffi::{c_char, CStr};
use std::slice;

use ar_drivers::connection::Connection;
use ar_drivers::ffi::{GetEuler, StartConnection, StopConnection};
use ar_drivers::fusion::Fusion;
use nalgebra::Vector3;

fn main() {
    let mut error: *const c_char = std::ptr::null();

    // stress test to ensure connection after disconnection works
    for _a in 0..100 {
        let code = StartConnection(&mut error);
        println!("starting, code {}", code);
        if !error.is_null() {
            // SAFETY: the message is valid until the next module call
            let message = unsafe { CStr::from_ptr(error) };
            println!("error: {}", message.to_string_lossy());
        }

        {
            let _serial = Connection::read_fusion(&mut |fusion| {
                let serial = fusion.glasses().serial().unwrap();
                println!("Got glasses, serial={}", serial);
            });
        }

        println!("");

        for _i in 0..500 {
            // let quaternion = {
            //     let ptr = GetQuaternion(&mut error);
            //     // Array::from_raw_parts(ptr, 4);
            //     unsafe { slice::from_raw_parts(ptr, 4) }
            // };
            let frd = {
                let ptr = GetEuler(&mut error);
                unsafe {
                    let slice = slice::from_raw_parts(ptr, 3);
                    Vector3::new(slice[0], slice[1], slice[2])
                }
            };
            println!("euler:\t{:10.7}(i={})", frd.transpose(), _i);
            // println!("quaternion:\t{:?}\teuler:\t{:?}", quaternion, euler);
        }

        let code = StopConnection(&mut error);
        println!("\nstopping, code {}", code);
        if !error.is_null() {
            // SAFETY: the message is valid until the next module call
            let message = unsafe { CStr::from_ptr(error) };
            println!("error: {}", message.to_string_lossy());
        }
    }
}
