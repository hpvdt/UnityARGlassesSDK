/*
magnetometer calibration: sample retention, online ellipsoid fitting, and live
quality assessment for the fusion pipeline
*/
mod bad_mag_cause;
pub use bad_mag_cause::{BadCalibration, BadMagCause, BadReading};

mod calibration_quality;
pub use calibration_quality::CalibrationQuality;
mod mag_calibrator;
pub use mag_calibrator::{MagCalibrationResult, MagCalibrator};
mod mag_model;
mod mag_samples;
mod sample_stats;
