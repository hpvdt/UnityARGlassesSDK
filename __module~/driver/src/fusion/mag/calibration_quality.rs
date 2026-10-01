/// Live calibration quality record of the current working candidate.
/// `coverage` is bounded in `[0, 1]`; the three loss fields are the raw
/// (unbounded) loss values of the online optimizer's objective over the
/// retained cache rows — lower is better — saved when the quality is
/// recomputed and reported together as the quality half of
/// [`MagCalibrationResult`](super::mag_calibrator::MagCalibrationResult).
/// `confidence` is derived from `coverage` alone; `radial_loss`,
/// `regularization_loss`, and `gravity_loss` are diagnostic records and
/// do not enter the confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationQuality {
    /// Directional coverage factor of the confidence in `[0, 1]`: the
    /// E-optimality score of the retained mean-centered unit directions.
    pub coverage: f32,
    /// Radial online objective of the working parameters over the retained
    /// cache rows,
    /// $J_r = \frac{1}{2 n} \sum_i e_{r,i}^2 + \frac{\lambda}{2} \|Q - c I\|_F^2$
    /// with $e_{r,i} = \phi(u_i)^T \theta - 1$ — the same radial loss the
    /// online optimizer minimizes, including the shape regularization term
    /// reported separately in `regularization_loss`. Unbounded; lower is
    /// better.
    pub radial_loss: f32,
    /// Shape-regularization loss of the working parameters,
    /// $\frac{\lambda}{2} \|Q - c I\|_F^2$. This is the prior term of the
    /// radial online objective that `radial_loss` reports; it depends only
    /// on the working coefficients, not on the retained rows, and is not
    /// bounded in `[0, 1]`. Zero once the working coefficients sit exactly
    /// on the prior.
    pub regularization_loss: f32,
    /// Gravity online objective over the retained rows carrying a gravity
    /// direction,
    /// $J_g = \frac{w_g}{2 n_g} \sum_i e_{g,i}^2$ with
    /// $e_{g,i} = (\psi(u_i, \tilde{g}_i)^T \theta - \kappa) / \sigma_g$ —
    /// the same gravity loss the online optimizer minimizes. Zero when
    /// gravity is disabled, the projection $\kappa$ is not yet seeded, or no
    /// retained row carries gravity: the objective then contains no gravity
    /// term, so an absent term and a perfect fit both report zero.
    /// Unbounded above; lower is better.
    pub gravity_loss: f32,
}

impl CalibrationQuality {
    pub(super) const ZERO: Self = Self {
        coverage: 0.0,
        radial_loss: 0.0,
        regularization_loss: 0.0,
        gravity_loss: 0.0,
    };

    pub(super) fn new(
        coverage: f32,
        radial_loss: f32,
        regularization_loss: f32,
        gravity_loss: f32,
    ) -> Self {
        Self {
            coverage,
            radial_loss,
            regularization_loss,
            gravity_loss,
        }
    }

    /// Current bounded calibration quality in `[0, 1]`: the clamped
    /// `coverage`. The loss diagnostics are deliberately excluded, so a
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
