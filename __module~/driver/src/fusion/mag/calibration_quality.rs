/// Live calibration quality factors of the current working candidate. The
/// sub-factors are bounded in `[0, 1]`; `regularization_loss` is the raw
/// (unbounded) regularization term of the radial objective, reported for
/// logging. Only the sub-factors and the loss are stored as state, reset
/// together, and reported together as the quality half of
/// [`MagCalibrationResult`](super::mag_calibrator::MagCalibrationResult);
/// `fitness` and `confidence` are derived from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationQuality {
    /// Directional coverage factor of the confidence in `[0, 1]`: the
    /// E-optimality score of the retained mean-centered unit directions.
    pub coverage: f32,
    /// Radial fitness sub-factor in `[0, 1]`: the bounded RMS-equivalent
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
    /// Gravity-consistency fitness sub-factor in `[0, 1]`: the bounded fit
    /// of the gravity-projection surrogate over the retained rows carrying
    /// a gravity direction. `1.0` while no retained row carries gravity or
    /// the gravity term is disabled, so a magnetometer-only stream is never
    /// penalized.
    pub gravity_fitness: f32,
    // /// Effective relative weight of the gravity fitness factor in
    // /// [`CalibrationQuality::fitness`]: the configured `gravity_weight`
    // /// while the gravity-projection statistic is live, `0.0` otherwise.
    // /// Zero reduces the combination to `radial_fitness` exactly, matching
    // /// the objective with the surrogate disabled.
    // pub gravity_term_weight: f32, /*$w_g$*/
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

    // /// Combined fitness factor of the confidence in `[0, 1]`: the weighted
    // /// arithmetic mean
    // /// `(radial_fitness + w * gravity_fitness) / (1 + w)` with
    // /// `w = gravity_term_weight`. The online objective *adds* its two data
    // /// terms with relative weight `w_g` (each normalized by its own
    // /// observation count, so counts cancel), so the combination must add
    // /// too: any multiplicative form — a plain product or a weighted power
    // /// mean — gives each factor a veto the objective does not have (at
    // /// `w_g = 0.01` a completely broken gravity surrogate costs the
    // /// objective about 1%, while geometrically it would zero the fitness).
    // /// The per-factor ramps (and hence the gravity residual floor) stay
    // /// applied per statistic before combination, so the surrogate's
    // /// transient dip inconsistency while the preconditioner frame converges
    // /// never leaks into the radial assessment. `w = 0` —
    // /// gravity disabled, unseeded, or absent from the cache — reduces the
    // /// combination to `radial_fitness`, and a gravity-free stream is never
    // /// penalized.
    // pub fn fitness(&self) -> f32 {
    //     let weight = self.gravity_term_weight;
    //     if weight.is_finite() && weight > 0.0 {
    //         (self.radial_fitness + weight * self.gravity_fitness) / (1.0 + weight)
    //     } else {
    //         self.radial_fitness
    //     }
    // }

    /// Current bounded calibration quality in `[0, 1]`: the clamped product
    /// of `coverage` and `fitness`.
    pub fn confidence(&self) -> f32 {
        let quality = self.coverage;
        // let quality = self.coverage * self.fitness();
        if quality.is_finite() {
            quality.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
