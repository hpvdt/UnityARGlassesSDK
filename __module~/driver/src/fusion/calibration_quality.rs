/// Live calibration quality factors of the current working candidate, all
/// bounded in `[0, 1]`. Only the sub-factors are stored as state, reset
/// together, and reported together as the quality half of
/// [`MagCalibrationResult`](super::mag_calibrator::MagCalibrationResult);
/// `fitness` and `confidence` are derived from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationQuality {
    /// Directional coverage factor of the confidence in `[0, 1]`: the
    /// E-optimality score of the retained mean-centered unit directions.
    pub coverage: f32,
    /// Radial fitness sub-factor in `[0, 1]`: the bounded mean-square
    /// algebraic ellipsoid residual `phi(u_i)^T theta - 1` of the working
    /// parameters over the retained cache rows — the same data term the
    /// online optimizer minimizes.
    pub radial_fitness: f32,
    /// Gravity-consistency fitness sub-factor in `[0, 1]`: the bounded fit
    /// of the gravity-projection surrogate over the retained rows carrying
    /// a gravity direction. `1.0` while no retained row carries gravity or
    /// the gravity term is disabled, so a magnetometer-only stream is never
    /// penalized.
    pub gravity_fitness: f32,
}

impl CalibrationQuality {
    pub(super) const ZERO: Self = Self {
        coverage: 0.0,
        radial_fitness: 0.0,
        gravity_fitness: 0.0,
    };

    pub(super) fn new(coverage: f32, radial_fitness: f32, gravity_fitness: f32) -> Self {
        Self {
            coverage,
            radial_fitness,
            gravity_fitness,
        }
    }

    /// Combined fitness factor of the confidence in `[0, 1]`:
    /// `radial_fitness * gravity_fitness`.
    pub fn fitness(&self) -> f32 {
        self.radial_fitness * self.gravity_fitness
    }

    /// Current bounded calibration quality in `[0, 1]`: the clamped product
    /// of `coverage` and `fitness`.
    pub fn confidence(&self) -> f32 {
        let quality = self.coverage * self.fitness();
        if quality.is_finite() {
            quality.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
