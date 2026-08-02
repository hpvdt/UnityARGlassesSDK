// Copyright (C) 2023, Alex Badics
// This file is part of ar-drivers-rs
// Licensed under the MIT license. See LICENSE file in the project root for details.

#![warn(missing_docs)]
//! This crate contains a simplified driver for Rokid Air and Mad Gaze Glow AR glasses.
//! It supports getting basic sensor data and setting up the display.
//!
//! Example usage (in a thread, probably):
//! ```ignore
//! let mut glasses = any_glasses().unwrap();
//! loop {
//!     match glasses.read_event().unwrap() {
//!         GlassesEvent::AccGyro {accelerometer, gyroscope} => ...,
//!         GlassesEvent::Magnetometer(data) => ...,
//!         GlassesEvent::KeyPress(data) => ...,
//!         _ => {}
//!     }
//! }
//! ```
//!
//! As opposed to e.g. Rokid's own API, this is all that you get, since this is what comes
//! out of the hardware. To get quaternions, you should probably use a lib that implements
//! Madgwicks algorithm or a proper EKF. One good choice is the `eskf` crate.
//!
//! ## Feature flags
//!
//! Support for individual AR glasses types ca be enabled with the following features:
//!
//! * `mad_gaze`: Mad Gaze Glow
//! * `xreal`: XREAL Light
//! * `rokid`: Rokid Air
//!
//! All of them are enabled by default, which may bring in some unwanted dependencies if you
//! only want to support a specific type.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector2, Vector3};

/// Sensor fusion implementations and AHRS reference-frame helpers.
pub mod fusion;

#[cfg(feature = "grawoow")]
pub mod grawoow;
#[cfg(feature = "mad_gaze")]
pub mod mad_gaze;
#[cfg(feature = "rokid")]
pub mod rokid;
#[cfg(feature = "xreal")]
pub mod xreal_air;
#[cfg(feature = "xreal")]
pub mod xreal_light;

/// Singleton connection that runs sensor fusion in a background thread.
pub mod connection;

/// C ABI entry points for the Unity integration.
pub mod ffi;

pub mod sim;
mod util;

pub use sim::SimMotion;

/// Possible errors resulting from `ar-drivers` API calls
#[derive(Debug)]
pub enum Error {
    /// A standard IO error happened. See [`std::io::Error`] for specifics
    IoError(std::io::Error),
    /// An rusb error happened. See [`rusb::Error`] for specifics
    #[cfg(feature = "rusb")]
    UsbError(rusb::Error),
    /// A hidapi error happened. See [`hidapi::HidError`] for specifics
    #[cfg(feature = "hidapi")]
    HidError(hidapi::HidError),
    /// A serialport error happened. See [`serialport::Error`] for specifics
    #[cfg(feature = "serialport")]
    SerialPortError(serialport::Error),
    /// No glasses were found.
    NotFound,
    /// The feature is not available with this headset.
    NotImplemented,
    /// Packet sending or reception timed out. Note that this is not the only
    /// timeout error that is sent (e.g. UsbError can contain a timeout), and
    /// also this is usually a fatal one.
    PacketTimeout,
    /// Other fatal error, usually a problem with the library itself, or
    /// a device support issue. File a bug if you encounter this.
    Other(&'static str),
    /// Cannot get mutex lock
    ConcurrencyError,
}

type Result<T> = std::result::Result<T, Error>;

type Rw<T> = Arc<Mutex<T>>;

fn rw<T>(v: T) -> Rw<T> {
    Arc::new(Mutex::new(v))
}

fn rw_write<T>(v: &Rw<T>) -> std::sync::MutexGuard<'_, T> {
    v.lock().unwrap()
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IoError(e) => Some(e),
            #[cfg(feature = "rusb")]
            Error::UsbError(e) => Some(e),
            #[cfg(feature = "hidapi")]
            Error::HidError(e) => Some(e),
            #[cfg(feature = "serialport")]
            Error::SerialPortError(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::IoError(_) => "I/O error",
            #[cfg(feature = "rusb")]
            Error::UsbError(_) => "Libusb error",
            #[cfg(feature = "hidapi")]
            Error::HidError(_) => "Hidapi error",
            #[cfg(feature = "serialport")]
            Error::SerialPortError(_) => "Serial error",
            Error::ConcurrencyError => "concurrency error",
            Error::NotFound => "Glasses not found",
            Error::NotImplemented => "Not implemented for these glasses",
            Error::PacketTimeout => "Packet timeout",
            Error::Other(s) => s,
        })
    }
}

/// AR glasses sensor event, got from [`ARGlasses::read_event`]
///
/// Coordinate system is "RUB": Positive X is Right, Positive Y is Up, Positive Z is backwards.
/// This is the same as the Android sensor coordinate system.
#[derive(derive_more::Debug, Clone)]
pub enum GlassesEvent {
    /// Synchronized accelerometer and gyroscope data.
    AccGyro {
        /// Accelerometer data in m^2/s.
        ///
        /// Remember that while gravitational acceleration is "down", the acceleration
        /// the device "feels" is the one opposite from that, so the normal reading
        /// when the device is upright is (0, 9.81, 0)
        #[debug(
            "[x={:+10.4}, y={:+10.4}, z={:+10.4}]",
            accelerometer.x,
            accelerometer.y,
            accelerometer.z
        )]
        accelerometer: Vector3<f32>,
        /// Gyroscope data. Right handed rotation in rad/sec,
        /// e.g. turning left is positive y axis.
        /// from left to right:
        /// pitch, yaw, roll
        /// Right, Up, Back (RUB)
        #[debug(
            "[x={:+10.4}, y={:+10.4}, z={:+10.4}]",
            gyroscope.x,
            gyroscope.y,
            gyroscope.z
        )]
        gyroscope: Vector3<f32>,
        /// Timestamp, in device time, in microseconds
        timestamp: u64,
    },
    /// Magnetometer data.
    Magnetometer {
        /// Direction of magnetic north (more or less). Unit is uT.
        #[debug(
            "[x={:+10.4}, y={:+10.4}, z={:+10.4}]",
            magnetometer.x,
            magnetometer.y,
            magnetometer.z
        )]
        magnetometer: Vector3<f32>,
        /// Timestamp, in device time, in microseconds
        timestamp: u64,
    },
    /// A key was pressed (sent once per press)
    /// The number is a key ID, starting from 0.
    KeyPress(u8),

    /// Proximity sensor senses the user, i.e. the glasses were put on
    /// Sent once per event.
    ProximityNear,

    /// Proximity sensor senses the user, i.e. the glasses were taken off.
    /// Sent once per event.
    ProximityFar,
    /// Ambient light level. Unit is vendor-specific
    AmbientLight(u16),
    /// V-sync happened on the device
    VSync,
}

/// Display mode used by [`ARGlasses::set_display_mode`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Picture should be same for both eyes (simple full HD mode)
    SameOnBoth,
    /// Set display to 3840*1080 or 3840x1200,
    /// where the left half is the left eye, the right half is the right eye
    Stereo,
    /// Set display to half-SBS mode, which presents itself as 1920*1080 resolution,
    /// but actually scales down everything to 960x540,then upscales to 3840x1080
    HalfSBS,
    /// Set display to mirrored high refresh rate mode (typically 120Hz)
    HighRefreshRate,
    /// Set display to high refresh rate SBS mode
    HighRefreshRateSBS,
}

/// Display side used by [`ARGlasses::view_matrix`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Left display
    Left,
    /// Right display
    Right,
}

/// Common interface for AR implemented glasses
pub trait ARGlasses: Send {
    /// Get the serial number of the glasses
    fn serial(&mut self) -> Result<String>;
    /// Get a single sensor event. Blocks.
    fn read_event(&mut self) -> Result<GlassesEvent>;
    /// Start writing implementation-defined raw input packets to `path`.
    ///
    /// Drivers without packet-logging support treat this as a no-op.
    fn start_packet_logging(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
    /// Stop packet logging and flush all buffered data.
    ///
    /// Drivers without packet-logging support treat this as a no-op.
    fn stop_packet_logging(&mut self) -> Result<()> {
        Ok(())
    }
    /// Get the display mode of the glasses. See [`DisplayMode`]
    fn get_display_mode(&mut self) -> Result<DisplayMode>;
    /// Set the display mode of the glasses. See [`DisplayMode`]
    fn set_display_mode(&mut self, display_mode: DisplayMode) -> Result<()>;
    /// Field of view of the display along the horizontal axis, in radians
    fn display_fov(&self) -> f32;
    /// Transformation from IMU frame to display frame, at the specified
    /// IPD (interpupillary distance). The `ipd` parameter is in meters.
    /// A typical value is 0.07.
    fn imu_to_display_matrix(&self, side: Side, ipd: f32) -> Isometry3<f64>;
    /// Name of the device
    fn name(&self) -> &'static str;
    /// Get built-in camera descriptors
    fn cameras(&self) -> Result<Vec<CameraDescriptor>> {
        Ok(Vec::new())
    }
    /// Get the available display matrices
    fn display_matrices(&self) -> Result<(DisplayMatrices, DisplayMatrices)> {
        Err(Error::NotImplemented)
    }
    /// The additional delay (in usecs) of the glasses' display from getting the data
    /// on DisplayPort. This is not really an absolute value, but more of
    /// a relative measure between different glasses.
    /// In the future this may depend on the current display mode.
    fn display_delay(&self) -> u64;
}

/// Represents one built-in camera
///
/// Warning: Experimental. May change between any versions.
#[derive(Debug, Clone)]
pub struct CameraDescriptor {
    /// The unique name for the type of the camera.
    pub name: &'static str,
    /// The width and height in pixels for the calibration data
    pub resolution: Vector2<f64>,
    /// The intrinsic matrix of the camera
    pub intrinsic_matrix: Matrix3<f64>,
    /// Distortion coefficients: k1, k2, p1, p2, k3
    pub distortion: [f64; 5],
    /// Additional rectification matrix for stereo cameras
    pub stereo_rotation: UnitQuaternion<f64>,
    /// Transformation from the IMU frame to the camera frame
    pub imu_to_camera: Isometry3<f64>,
}

/// Represents the decomposed extrinsic and intrinsic matrices for the display's lens.
#[derive(Debug, Clone)]
pub struct DisplayMatrices {
    /// The intrinsic matrix of the display
    pub intrinsic_matrix: Matrix3<f64>,
    /// Native resolution of the display, in pixels.
    pub resolution: (u32, u32),
    /// Extrinsic matrix of the display; translation in meters.
    pub isometry: Isometry3<f64>,
}

/// Detect and connect to any supported glasses, falling back to the simulated
/// [`Dummy`] device when no hardware is found.
pub fn any_glasses_or_dummy() -> Result<Box<dyn ARGlasses>> {
    any_glasses().or_else(|e| {
        println!("{} fall back to dummy glasses", e);

        Ok(Box::new(sim::Dummy {}))
    })
}

fn upcast<G: ARGlasses + 'static>(result: Result<G>) -> Result<Box<dyn ARGlasses>> {
    result.map(|glasses| Box::new(glasses) as Box<dyn ARGlasses>)
}

/// Convenience function to detect and connect to any of the supported glasses
#[cfg(not(target_os = "android"))]
pub fn any_glasses() -> Result<Box<dyn ARGlasses>> {
    let glasses_factories: Vec<(&str, fn() -> Result<Box<dyn ARGlasses>>)> = vec![
        #[cfg(feature = "rokid")]
        ("RokidAir", || upcast(rokid::RokidAir::new())),
        #[cfg(feature = "xreal")]
        ("XrealAir", || upcast(xreal_air::XrealAir::new())),
        #[cfg(feature = "xreal")]
        ("XrealLight", || upcast(xreal_light::XrealLight::new())),
        #[cfg(feature = "grawoow")]
        ("GrawoowG530", || upcast(grawoow::GrawoowG530::new())),
        #[cfg(feature = "mad_gaze")]
        ("MadGazeGlow", || upcast(mad_gaze::MadGazeGlow::new())),
    ];

    glasses_factories
        .into_iter()
        .find_map(|(glasses_type, factory)| {
            let factory: fn() -> Result<Box<dyn ARGlasses>> = factory;

            factory()
                .map_err(|e| {
                    //
                    println!("can't find {}: {}", glasses_type, e)
                })
                .ok()
                .map(|v| {
                    println!("found {}", glasses_type);
                    v
                })
        })
        .ok_or(Error::NotFound)
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}

#[cfg(feature = "rusb")]
impl From<rusb::Error> for Error {
    fn from(e: rusb::Error) -> Self {
        Error::UsbError(e)
    }
}

#[cfg(feature = "hidapi")]
impl From<hidapi::HidError> for Error {
    fn from(e: hidapi::HidError) -> Self {
        Error::HidError(e)
    }
}

#[cfg(feature = "serialport")]
impl From<serialport::Error> for Error {
    fn from(e: serialport::Error) -> Self {
        Error::SerialPortError(e)
    }
}

impl<T> From<PoisonError<T>> for Error {
    fn from(_e: PoisonError<T>) -> Self {
        Error::ConcurrencyError
    }
}

impl From<&'static str> for Error {
    fn from(e: &'static str) -> Self {
        Error::Other(e)
    }
}
