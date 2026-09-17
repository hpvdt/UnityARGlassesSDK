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
    /// Effective relative weight of the gravity fitness factor in
    /// [`CalibrationQuality::fitness`]: the configured `gravity_weight`
    /// while the gravity-projection statistic is live, `0.0` otherwise.
    /// Zero reduces the combination to `radial_fitness` exactly, matching
    /// the objective with the surrogate disabled.
    pub gravity_term_weight: f32,
}

impl CalibrationQuality {
    pub(super) const ZERO: Self = Self {
        coverage: 0.0,
        radial_fitness: 0.0,
        gravity_fitness: 0.0,
        // FIXME: this is wrong, if gravity vector exists, it should be used for optimisation.
        //   (to be elaborated)
        //  Investigate the current optimisation loss function to ensure that:
        //  ...
        gravity_term_weight: 0.0,
    };

    pub(super) fn new(
        coverage: f32,
        radial_fitness: f32,
        gravity_fitness: f32,
        gravity_term_weight: f32,
    ) -> Self {
        Self {
            coverage,
            radial_fitness,
            gravity_fitness,
            gravity_term_weight,
        }
    }

    /// Combined fitness factor of the confidence in `[0, 1]`: the weighted
    /// arithmetic mean
    /// `(radial_fitness + w * gravity_fitness) / (1 + w)` with
    /// `w = gravity_term_weight`. The online objective *adds* its two data
    /// terms with relative weight `w_g` (each normalized by its own
    /// observation count, so counts cancel), so the combination must add
    /// too: any multiplicative form — a plain product or a weighted power
    /// mean — gives each factor a veto the objective does not have (at
    /// `w_g = 0.01` a completely broken gravity surrogate costs the
    /// objective about 1%, while geometrically it would zero the fitness).
    /// The per-factor ramps (and hence the gravity bias floor) stay applied
    /// per statistic before combination, so the surrogate's known
    /// anisotropic bias never leaks into the radial assessment. `w = 0` —
    /// gravity disabled, unseeded, or absent from the cache — reduces the
    /// combination to `radial_fitness`, and a gravity-free stream is never
    /// penalized.
    pub fn fitness(&self) -> f32 {
        let weight = self.gravity_term_weight;
        if weight.is_finite() && weight > 0.0 {
            (self.radial_fitness + weight * self.gravity_fitness) / (1.0 + weight)
        } else {
            self.radial_fitness
        }
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
