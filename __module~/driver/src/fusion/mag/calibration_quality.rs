/// Live calibration quality factors of the current working candidate. The
/// sub-factors are bounded in `[0, 1]`; `regularization_loss` is the raw
/// (unbounded) regularization term of the radial objective, reported for
/// logging. Only the sub-factors and the loss are stored as state, reset
/// together, and reported together as the quality half of
/// [`MagCalibrationResult`](super::mag_calibrator::MagCalibrationResult).
/// `confidence` is derived from `coverage` alone; `radial_fitness`,
/// `gravity_fitness`, and `regularization_loss` are diagnostic records and
/// do not enter the confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationQuality {
    /// Directional coverage factor of the confidence in `[0, 1]`: the
    /// E-optimality score of the retained mean-centered unit directions.
    pub coverage: f32,
    /// Radial fitness diagnostic in `[0, 1]`: the bounded RMS-equivalent
    /// score of the full radial online objective
    /// $J_r = \frac{1}{2 n} \sum_i e_{r,i}^2 + \frac{\lambda}{2} \|Q - c I\|_F^2$
    /// of the working parameters over the retained cache rows — the same
    /// loss the online optimizer minimizes, including the shape
    /// regularization term reported separately in `regularization_loss`.
    pub radial_fitness: f32,
    /// Shape-regularization loss of the working parameters,
    /// $\frac{\lambda}{2} \|Q - c I\|_F^2$. This is the prior term of the
    /// radial online objective that `radial_fitness` scores; it depends only
    /// on the working coefficients, not on the retained rows, and is not
    /// bounded in `[0, 1]`. Zero once the working coefficients sit exactly
    /// on the prior.
    pub regularization_loss: f32,
    /// Gravity-consistency fitness diagnostic in `[0, 1]`: the bounded fit
    /// of the gravity-projection surrogate over the retained rows carrying
    /// a gravity direction. `1.0` while no retained row carries gravity or
    /// the gravity term is disabled, marking the statistic as neutral.
    pub gravity_fitness: f32,
}

impl CalibrationQuality {
    pub(super) const ZERO: Self = Self {
        coverage: 0.0,
        radial_fitness: 0.0,
        regularization_loss: 0.0,
        gravity_fitness: 0.0,
    };

    pub(super) fn new(
        coverage: f32,
        radial_fitness: f32,
        regularization_loss: f32,
        gravity_fitness: f32,
    ) -> Self {
        Self {
            coverage,
            radial_fitness,
            regularization_loss,
            gravity_fitness,
        }
    }

    /// Current bounded calibration quality in `[0, 1]`: the clamped
    /// `coverage`. The fitness diagnostics are deliberately excluded, so a
    /// transiently poor fit does not mask good directional support.
    pub fn confidence(&self) -> f32 {
        let quality = self.coverage;
        if quality.is_finite() {
            quality.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
