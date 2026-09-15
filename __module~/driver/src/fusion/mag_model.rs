use nalgebra::{SMatrix, SVector, Vector3};

use super::CalibrationQuality;

/// Number of ellipsoid coefficients fitted by the magnetometer calibration
/// model.
pub(super) const CALIBRATION_PARAMETER_COUNT: usize = 9;

/// Calibration model state behind `MagCalibrator`: the retained magnetometer
/// sample cache the quality statistics are estimated from, the online
/// ellipsoid coefficients with the sample normalization they are expressed
/// in, the learned gravity-projection state of the optional surrogate, and
/// the live quality factors derived from all of the above. Grouping the
/// fields keeps the quality-estimation inputs (`update_quality`) together
/// and separate from the optimizer bookkeeping, diversity neighbor cache,
/// and publication state that the calibrator owns itself.
pub(super) struct MagModel<const N: usize> {
    pub(super) sample_matrix: SMatrix<f32, N, 3>,
    pub(super) gravity_directions: [Option<Vector3<f32>>; N],
    pub(super) sample_row_count: usize,
    pub(super) parameters: SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    /// Sample mean $\mu$ of the retained magnetometer samples: the center of
    /// the sample normalization $u_i = (x_i - \mu) / r$.
    pub(super) sample_mean: Vector3<f32>,
    /// RMS radius $r$ of the retained magnetometer samples: the scale of the
    /// sample normalization $u_i = (x_i - \mu) / r$.
    pub(super) sample_rms_radius: f32,
    //FIXME, both sample_normalization_initialized and learned_gravity_projection_initialized are not required
    // whether the calibration is initialised should be totally determined by the confidence score of CalibrationQuality
    /// Whether the sample normalization $(\mu, r)$ of the retained
    /// magnetometer samples is usable: both finite and the radius above
    /// `f32::EPSILON`, which requires two distinct samples.
    pub(super) sample_normalization_initialized: bool,
    /// Learned scalar $\kappa$ of the gravity surrogate: the projection of
    /// the normalized gravity direction $g_i$ onto the ellipsoid normal
    /// $n_i = Q u_i + q / 2$ at a retained row, $\kappa = \psi(u_i, g_i)^T
    /// \theta$, which the surrogate keeps approximately constant across
    /// rows. Initialized once from the first usable observation and then
    /// refined by the optimizer gradient steps.
    pub(super) learned_gravity_projection: f32,
    /// Whether `learned_gravity_projection` holds a usable finite value.
    pub(super) learned_gravity_projection_initialized: bool,
    pub(super) gravity_weight: f32,
    /// Live calibration quality factors of the current working candidate,
    /// reset together with the model minimum and recomputed by
    /// `update_quality` on every publication evaluation.
    pub(super) quality: CalibrationQuality,
}
