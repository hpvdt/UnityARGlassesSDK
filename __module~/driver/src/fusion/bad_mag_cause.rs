/// Reason a magnetometer vector could not produce a calibrated FRD reading.
#[derive(Clone, Copy, derive_more::Debug, derive_more::From, PartialEq)]
pub enum BadMagCause {
    /// The magnetometer calibration is not usable.
    BadCalibration(BadCalibration),
    /// The magnetometer reading is not usable.
    BadReading(BadReading),
}

/// Reason a magnetometer calibration is not usable.
#[derive(Clone, Copy, derive_more::Debug, PartialEq)]
pub enum BadCalibration {
    /// Calibration optimization or candidate conversion produced unusable parameters.
    Unsolveable {
        /// Original rejection message.
        message: &'static str,
    },
    /// The samples or fitted soft-iron correction are numerically degenerate.
    DegenerateSoftIronMatrix {
        /// Estimated condition number of the fitted soft-iron correction.
        #[debug("{:+10.4}", condition)]
        condition: f32,
        /// Maximum accepted condition number.
        #[debug("{:+10.4}", max_condition)]
        max_condition: f32,
    },
}

/// Reason a magnetometer reading is not usable.
#[derive(Clone, Copy, derive_more::Debug, PartialEq)]
pub enum BadReading {
    /// The raw magnetometer vector is too small to be useful. TODO: this is actually not a problem as hard-iron zero can be very far
    // WeakRawReading {
    //     /// Actual raw vector norm.
    //     norm: f32,
    //     /// Minimum accepted vector norm.
    //     min_norm: f32,
    // },
    /// The calibrated vector is too small to be useful.
    WeakCalibratedReading {
        /// Actual calibrated vector norm.
        #[debug("{:+10.4}", norm)]
        norm: f32,
        /// Minimum accepted vector norm.
        #[debug("{:+10.4}", min_norm)]
        min_norm: f32,
    },
}
