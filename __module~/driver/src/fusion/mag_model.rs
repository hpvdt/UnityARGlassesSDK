use nalgebra::{SMatrix, SVector, Vector3};

use super::CalibrationQuality;

/// Number of ellipsoid coefficients fitted by the magnetometer calibration
/// model.
pub(super) const CALIBRATION_PARAMETER_COUNT: usize = 9;

/// Calibration model state behind `MagCalibrator`: the retained sample cache
/// the quality statistics are estimated from, the online ellipsoid
/// coefficients with the cache normalization they are expressed in, the
/// gravity-projection state of the optional surrogate, and the live quality
/// factors derived from all of the above. Grouping the fields keeps the
/// quality-estimation inputs (`update_quality`) together and separate from
/// the optimizer bookkeeping,
/// diversity neighbor cache, and publication state that the calibrator owns
/// itself.
pub(super) struct MagModel<const N: usize> {
    pub(super) sample_matrix: SMatrix<f32, N, 3>,
    pub(super) gravity_directions: [Option<Vector3<f32>>; N],
    pub(super) sample_row_count: usize,
    pub(super) parameters: SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    pub(super) normalization_mean: Vector3<f32>,
    pub(super) normalization_radius: f32,
    pub(super) normalization_initialized: bool,
    pub(super) gravity_projection: f32,
    pub(super) gravity_projection_initialized: bool,
    pub(super) gravity_weight: f32,
    /// Live calibration quality factors of the current working candidate,
    /// reset together with the model minimum and recomputed by
    /// `update_quality` on every publication evaluation.
    pub(super) quality: CalibrationQuality,
}
