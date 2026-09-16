use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use super::bad_mag_cause::BadCalibration;
use super::mag_calibrator::SHAPE_PRIOR_SCALE;
use super::mag_samples::{MagSamples, Row};
use super::CalibrationQuality;

/// Number of ellipsoid coefficients fitted by the magnetometer calibration
/// model.
pub(super) const CALIBRATION_PARAMETER_COUNT: usize = 9;

/// Gram sum of the retained direction features, `sum_i varphi(d_i)
/// varphi(d_i)^T`, backing the coverage score.
pub(super) type CoverageGramMatrix =
    SMatrix<f32, CALIBRATION_PARAMETER_COUNT, CALIBRATION_PARAMETER_COUNT>;
const MAX_CORRECTION_CONDITION: f32 = 1.0e1;
const MAX_RADIAL_RMS: f32 = 0.1;
/// Gravity-projection RMS residual below which the gravity fitness is 1.
/// The normal-projection surrogate is biased under anisotropic soft iron, so
/// even a perfect fit keeps an irreducible residual; the floor keeps that
/// bias from dragging down a good calibration.
const GRAVITY_RMS_FLOOR: f32 = 0.1;
/// Gravity-projection RMS residual at which the gravity fitness reaches 0,
/// ramping linearly down from 1 at `GRAVITY_RMS_FLOOR`. Residuals live in
/// normalized ellipsoid-equation units; both constants are calibrated
/// against the synthetic consistent/contradictory gravity test. The
/// statistic is the mean square over the whole retained cache, whose
/// steady-state consistent-fit RMS sits above the trailing-window estimate
/// the original 0.3 ceiling was tuned against, so the ceiling is raised to
/// 0.35 to keep the same factor for the same physical fit quality
/// (SimMotion regression-validated).
const MAX_GRAVITY_RMS: f32 = 0.35;
/// Uniform-sphere reference for directional coverage: the smallest
/// eigenvalue of `E[varphi(d) varphi(d)^T]` over uniformly distributed unit
/// directions, where `varphi` is the coverage-feature vector with
/// `sqrt(2)` cross-term weights (see `coverage_feature`). A fully isotropic
/// cache scores 1 against this reference.
const COVERAGE_LAMBDA_REF: f32 = 2.0 / 15.0;

/// Finite hard-iron offset and soft-iron correction derived from the
/// working ellipsoid state of a [`MagModel`], ready to be published by the
/// calibrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CalibrationCandidate {
    pub(super) offset: Vector3<f32>,
    pub(super) correction: Matrix3<f32>,
}

/// Calibration model state behind `MagCalibrator`: the retained magnetometer
/// sample cache the quality statistics are estimated from, the online
/// ellipsoid coefficients with the sample normalization they are expressed
/// in, the learned gravity-projection state of the optional surrogate, and
/// the live quality factors derived from all of the above. Grouping the
/// fields keeps the quality-estimation inputs (`update_quality`) together
/// and separate from the optimizer bookkeeping, diversity neighbor cache,
/// and publication state that the calibrator owns itself.
pub(super) struct MagModel<const N: usize> {
    /// Retained magnetometer sample cache: the raw samples and the optional
    /// gravity direction carried by each row.
    pub(super) samples: MagSamples<N>,
    pub(super) sample_row_count: usize,
    pub(super) parameters: SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    /// Raw first moment of the retained magnetometer samples, maintained
    /// incrementally on append, replacement, and expiry. Backs
    /// `raw_mean_and_covariance` and thus `refresh_normalization`.
    pub(super) raw_sample_sum: Vector3<f64>,
    /// Raw second outer-product moment of the retained magnetometer
    /// samples; see `raw_sample_sum`.
    pub(super) raw_outer_product_sum: Matrix3<f64>,
    /// Sample mean $\mu$ of the retained magnetometer samples: the center of
    /// the sample normalization $u_i = (x_i - \mu) / r$.
    pub(super) sample_mean: Vector3<f32>,
    /// RMS radius $r$ of the retained magnetometer samples: the scale of the
    /// sample normalization $u_i = (x_i - \mu) / r$.
    pub(super) sample_rms_radius: f32,
    //FIXME, both sample_normalization_initialized and learned_gravity_projection_initialized are major vulnerability and should be removed
    // since Quality/confidence estimation relies on them. An uninitialised state entails a defective confidence score and premature output of corrected data, leading to aircraft crash
    // revise the code such that they are initialised from the beginning
    // run mag_calibrator_sim_motion test, make sure that any post-warmup confidence score contains a valid gravity fitness score
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

impl<const N: usize> MagModel<N> {
    pub(super) fn normalized_sample(&self, sample: Vector3<f32>) -> Vector3<f32> {
        (sample - self.sample_mean) / self.sample_rms_radius
    }

    pub(super) fn add_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum += sample;
        self.raw_outer_product_sum += sample * sample.transpose();
    }

    pub(super) fn remove_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum -= sample;
        self.raw_outer_product_sum -= sample * sample.transpose();
    }

    pub(super) fn clear_raw_moments(&mut self) {
        self.raw_sample_sum = Vector3::zeros();
        self.raw_outer_product_sum = Matrix3::zeros();
    }

    pub(super) fn raw_mean_and_covariance(&self) -> Option<(Vector3<f32>, Matrix3<f32>)> {
        if self.sample_row_count == 0 {
            return None;
        }
        let count = self.sample_row_count as f64;
        let mean = self.raw_sample_sum / count;
        let covariance = self.raw_outer_product_sum / count - mean * mean.transpose();
        let covariance = 0.5 * (covariance + covariance.transpose());
        let mean = mean.cast::<f32>();
        let covariance = covariance.cast::<f32>();
        if mean.iter().all(|value| value.is_finite())
            && covariance.iter().all(|value| value.is_finite())
        {
            Some((mean, covariance))
        } else {
            None
        }
    }

    /// Recomputes the current cache normalization from the raw moments
    /// without touching the working state. Every append, replacement, and
    /// expiry drift the mean and radius; the working coefficients keep
    /// their meaning in the new normalization directly, because the drift
    /// per cache mutation is `O(1 / sample_row_count)` and the online optimizer
    /// is already designed to track the moving convex optimum as cache
    /// replacements improve coverage. Working state is therefore never
    /// rebased or reset: only a zero-radius (empty or single-point) cache
    /// marks the normalization uninitialized, which keeps the optimizer
    /// idle until two distinct samples exist and reports quality zero
    /// through the usual unusable-candidate path.
    pub(super) fn refresh_normalization(&mut self) {
        let Some((sample_mean, covariance)) = self.raw_mean_and_covariance() else {
            self.sample_mean = Vector3::zeros();
            self.sample_rms_radius = 0.0;
            self.sample_normalization_initialized = false;
            return;
        };
        let radius = covariance.trace().sqrt();
        self.sample_normalization_initialized = sample_mean.iter().all(|value| value.is_finite())
            && radius.is_finite()
            && radius > f32::EPSILON;
        self.sample_mean = sample_mean;
        self.sample_rms_radius = radius;
    }

    /// Prior coefficient vector of the working ellipsoid state: the
    /// shape-prior scale on the `Q` diagonal and zero elsewhere. It is both
    /// the initial value of `parameters` and the center of the shape
    /// regularizer.
    pub(super) fn parameter_prior() -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
        SVector::from_row_slice(&[
            SHAPE_PRIOR_SCALE,
            SHAPE_PRIOR_SCALE,
            SHAPE_PRIOR_SCALE,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
    }

    /// Radial feature vector of the normalized ellipsoid equation: the
    /// quadratic and linear terms whose dot product with `parameters` is the
    /// algebraic residual `phi^T theta - 1`.
    pub(super) fn features(sample: Vector3<f32>) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
        // TODO: use nalgebra outer-product and vector-view operations instead of elementwise feature construction
        SVector::from_row_slice(&[
            sample.x * sample.x,
            sample.y * sample.y,
            sample.z * sample.z,
            2.0 * sample.x * sample.y,
            2.0 * sample.x * sample.z,
            2.0 * sample.y * sample.z,
            sample.x,
            sample.y,
            sample.z,
        ])
    }

    /// Initializes the learned gravity projection once from the first usable
    /// observation: the current one when it carries a gravity direction,
    /// otherwise the first retained row that does.
    pub(super) fn initialize_gravity_projection(
        &mut self,
        current_sample: Vector3<f32>,
        current_gravity: Option<Vector3<f32>>,
    ) {
        if self.learned_gravity_projection_initialized || self.gravity_weight == 0.0 {
            return;
        }
        let observation = current_gravity
            .map(|gravity| (current_sample, gravity))
            .or_else(|| {
                (0..self.sample_row_count).find_map(|row| {
                    let row = self.samples.view(row);
                    row.gravity().map(|gravity| (row.sample(), gravity))
                })
            });
        if let Some((sample, gravity)) = observation {
            let features = Self::gravity_features(self.normalized_sample(sample), gravity);
            let projection = features.dot(&self.parameters);
            if projection.is_finite() {
                self.learned_gravity_projection = projection;
                self.learned_gravity_projection_initialized = true;
            }
        }
    }

    /// Unpacks the packed coefficient vector `parameters` into the shape
    /// matrix `Q` and the linear coefficient vector `q` of the ellipsoid
    /// equation `u^T Q u + q^T u = 1`.
    pub(super) fn unpack_ellipsoid_coefficients(
        parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    ) -> (Matrix3<f32>, Vector3<f32>) {
        // Diagonal [Q00, Q11, Q22] followed by packed off-diagonal [Q01, Q02, Q12].
        let shape = Matrix3::from_fn(|row, col| {
            let index = if row == col { row } else { row + col + 2 };
            parameters[index]
        });
        (shape, parameters.fixed_rows::<3>(6).into_owned())
    }

    /// Features of the projection of gravity onto the ellipsoid normal
    /// `Q * sample + q / 2`. The projection is linear in the nine ellipsoid
    /// parameters, keeping the combined online objective convex and quadratic.
    pub(super) fn gravity_features(
        sample: Vector3<f32>,
        gravity: Vector3<f32>,
    ) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
        // TODO: use nalgebra outer-product and vector-view operations instead of elementwise feature construction
        SVector::from_row_slice(&[
            gravity.x * sample.x,
            gravity.y * sample.y,
            gravity.z * sample.z,
            gravity.x * sample.y + gravity.y * sample.x,
            gravity.x * sample.z + gravity.z * sample.x,
            gravity.y * sample.z + gravity.z * sample.y,
            0.5 * gravity.x,
            0.5 * gravity.y,
            0.5 * gravity.z,
        ])
    }

    /// Feature vector `varphi(d)` of a unit direction `d`: the term whose
    /// outer products `varphi(d) varphi(d)^T` build the coverage Gram matrix
    /// summed by `mean_centered_coverage` and scored by `coverage_from_gram`.
    /// The nine components are the ellipsoid-fit features with `sqrt(2)`
    /// cross-term weights; with that weighting the feature norm equals the
    /// rotation-invariant `tr(d d^T d d^T)`, so the induced rotation on
    /// feature space is orthogonal and the Gram eigenvalues are exactly
    /// rotation-invariant. Under the uniform spherical distribution
    /// `E[varphi varphi^T]` has eigenvalues `{1/3 x4, 2/15 x5}`; the
    /// smallest, `2/15`, is the `COVERAGE_LAMBDA_REF` uniform-sphere
    /// reference.
    pub(super) fn coverage_feature(
        direction: Vector3<f32>,
    ) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
        // TODO: use nalgebra outer-product and vector-view operations instead of elementwise feature construction
        SVector::<f32, CALIBRATION_PARAMETER_COUNT>::from_column_slice(&[
            direction.x * direction.x,
            direction.y * direction.y,
            direction.z * direction.z,
            std::f32::consts::SQRT_2 * direction.x * direction.y,
            std::f32::consts::SQRT_2 * direction.x * direction.z,
            std::f32::consts::SQRT_2 * direction.y * direction.z,
            direction.x,
            direction.y,
            direction.z,
        ])
    }

    /// E-optimality coverage of the retained directions: the smallest
    /// eigenvalue of the mean Gram matrix relative to the uniform-sphere
    /// reference. Rotation-invariant by construction, and a cache whose
    /// directions support fewer than nine independent features (for example
    /// near-planar motion) is rank-deficient and scores near zero.
    pub(super) fn coverage_from_gram(
        gram_sum: &CoverageGramMatrix,
        sample_row_count: usize,
    ) -> f32 {
        if sample_row_count < CALIBRATION_PARAMETER_COUNT {
            return 0.0;
        }
        let mean_gram = gram_sum / sample_row_count as f32;
        let lambda_min = SymmetricEigen::new(mean_gram).eigenvalues.min();
        (lambda_min / COVERAGE_LAMBDA_REF).clamp(0.0, 1.0)
    }

    /// Coverage of the retained rows, mean-centered and recomputed from the
    /// current cache on each quality update. Recomputing keeps every
    /// direction centered on the current cache mean, so no insertion-time
    /// snapshots, incremental Gram state, or drift-triggered rebuilds are
    /// needed. The cache mean is used rather than the fitted hard-iron
    /// offset: the offset's component along the thinnest data direction is
    /// itself unconstrained for near-planar support, which destabilizes the
    /// score exactly where it must be decisive. A near-planar cache stays
    /// rank-deficient under any centering.
    pub(super) fn mean_centered_coverage(&self) -> f32 {
        let mut gram_sum = CoverageGramMatrix::zeros();
        for row in 0..self.sample_row_count {
            let centered = self.samples.view(row).sample() - self.sample_mean;
            if let Some(direction) = centered.try_normalize(f32::EPSILON) {
                let feature = Self::coverage_feature(direction);
                gram_sum += feature * feature.transpose();
            }
        }
        Self::coverage_from_gram(&gram_sum, self.sample_row_count)
    }

    /// Radial fitness in `[0, 1]`: a linear ramp from 1 at zero RMS to 0 at
    /// `MAX_RADIAL_RMS`, applied to the mean square radial residual
    /// `||A (x_i - b)|| - 1` recomputed over the retained rows with the
    /// current candidate. A missing or unusable statistic scores 0.
    pub(super) fn radial_fitness_score(mean_square: Option<f32>) -> f32 {
        match mean_square {
            Some(mean_square) if mean_square.is_finite() && mean_square >= 0.0 => {
                (1.0 - mean_square.sqrt() / MAX_RADIAL_RMS).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    /// Gravity fitness in `[0, 1]`: a linear ramp from 1 at the
    /// `GRAVITY_RMS_FLOOR` residual to 0 at `MAX_GRAVITY_RMS`, applied to
    /// the mean square gravity-projection residual `psi^T theta - kappa`
    /// recomputed over the retained rows carrying a gravity direction.
    /// Unlike the radial score, a missing statistic maps to a neutral 1:
    /// gravity is optional, so an absent or disabled gravity term must
    /// never penalize a magnetometer-only calibration. The residual
    /// measures constancy of the ellipsoid-normal projection, which matches
    /// the corrected-direction dot product only for isotropic correction;
    /// the score inherits the surrogate's anisotropic soft-iron bias.
    pub(super) fn gravity_fitness_score(mean_square: Option<f32>) -> f32 {
        match mean_square {
            Some(mean_square) if mean_square.is_finite() && mean_square >= 0.0 => {
                let rms = mean_square.sqrt();
                ((MAX_GRAVITY_RMS - rms) / (MAX_GRAVITY_RMS - GRAVITY_RMS_FLOOR)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        }
    }

    fn condition_number(eigenvalues: &Vector3<f32>) -> f32 {
        let min = eigenvalues.min();
        let max = eigenvalues.max();
        if !min.is_finite() || !max.is_finite() || min <= 0.0 {
            f32::INFINITY
        } else {
            max / min
        }
    }

    /// Derives one finite SPD correction candidate from the current online
    /// ellipsoid state without scanning retained rows.
    pub(super) fn working_candidate(&self) -> Result<CalibrationCandidate, BadCalibration> {
        if !self.sample_normalization_initialized
            || !self.sample_mean.iter().all(|value| value.is_finite())
            || !self.sample_rms_radius.is_finite()
            || self.sample_rms_radius <= f32::EPSILON
        {
            return Err(BadCalibration::Unsolveable {
                message: "sample normalization is non-finite or zero",
            });
        }
        let parameters = self.parameters;
        if !parameters.iter().all(|value| value.is_finite()) {
            return Err(BadCalibration::Unsolveable {
                message: "online calibration produced non-finite parameters",
            });
        }

        let (shape, linear) = Self::unpack_ellipsoid_coefficients(&parameters);
        let shape_eigen = shape.symmetric_eigen();
        let correction_condition = Self::condition_number(&shape_eigen.eigenvalues).sqrt();
        if correction_condition > MAX_CORRECTION_CONDITION {
            return Err(BadCalibration::DegenerateSoftIronMatrix {
                condition: correction_condition,
                max_condition: MAX_CORRECTION_CONDITION,
            });
        }

        let shape_cholesky = shape
            .cholesky()
            .ok_or(BadCalibration::DegenerateSoftIronMatrix {
                condition: f32::INFINITY,
                max_condition: MAX_CORRECTION_CONDITION,
            })?;
        let normalized_offset = -0.5 * shape_cholesky.solve(&linear);
        let ellipsoid_scale = 1.0 + normalized_offset.dot(&(shape * normalized_offset));
        if !ellipsoid_scale.is_finite() || ellipsoid_scale <= f32::EPSILON {
            return Err(BadCalibration::Unsolveable {
                message: "ellipsoid normalization is non-positive",
            });
        }

        let square_root = Matrix3::from_diagonal(
            &shape_eigen
                .eigenvalues
                .map(|value| (value / ellipsoid_scale).sqrt()),
        );
        let correction =
            shape_eigen.eigenvectors * square_root * shape_eigen.eigenvectors.transpose()
                / self.sample_rms_radius;
        let offset = self.sample_mean + self.sample_rms_radius * normalized_offset;
        if !offset.iter().all(|value| value.is_finite())
            || !correction.iter().all(|value| value.is_finite())
        {
            return Err(BadCalibration::Unsolveable {
                message: "calibration produced non-finite parameters",
            });
        }

        Ok(CalibrationCandidate { offset, correction })
    }

    /// Updates the live quality of the current working candidate.
    /// Normalization uses the maintained raw moments; coverage and both
    /// fitness statistics rescan the retained rows, so an expired or
    /// replaced row stops contributing to the reported quality immediately.
    ///
    /// FIXME: the Air 1 replay shows block-long post-warm-up radial-fitness
    /// dips to zero even though the statistic is recomputed from the
    /// retained rows on every quality update and rows only leave the cache
    /// through expiry or replacement. Working state is never rebased or
    /// reset, so the dips are genuine working-candidate degradation over
    /// the retained rows on certain trace segments, not a statistic- or
    /// normalization-lifecycle artifact. Investigate why the online
    /// candidate degrades there instead of converging.
    pub(super) fn update_quality(&mut self) -> Option<CalibrationCandidate> {
        if self.sample_row_count < CALIBRATION_PARAMETER_COUNT {
            self.quality = CalibrationQuality::ZERO;
            return None;
        }
        let candidate = match self.working_candidate() {
            Ok(candidate) => candidate,
            Err(_) => {
                self.quality = CalibrationQuality::ZERO;
                return None;
            }
        };
        // Radial fitness: mean square of the corrected-radius residual over
        // every retained row, in fixed ascending row order so the result is
        // bit-deterministic. A non-finite accumulation marks the statistic
        // unusable, matching the zero-quality path above.
        let mut radial_square_sum = 0.0f32;
        for row in 0..self.sample_row_count {
            let residual = (candidate.correction
                * (self.samples.view(row).sample() - candidate.offset))
                .norm()
                - 1.0;
            radial_square_sum += residual * residual;
        }
        let radial_mean_square = radial_square_sum / self.sample_row_count as f32;
        if !radial_mean_square.is_finite() {
            self.quality = CalibrationQuality::ZERO;
            return None;
        }
        // Gravity fitness: mean square of the projection residual over the
        // retained rows that carry a gravity direction. The statistic stays
        // absent (neutral 1 below) when gravity is disabled, uninitialized,
        // or carried by no retained row.
        let mut gravity_square_sum = 0.0f32;
        let mut gravity_count = 0usize;
        let gravity_mean_square =
            if self.learned_gravity_projection_initialized && self.gravity_weight > 0.0 {
                for row in 0..self.sample_row_count {
                    let row = self.samples.view(row);
                    if let Some(gravity) = row.gravity() {
                        let residual =
                            Self::gravity_features(self.normalized_sample(row.sample()), gravity)
                                .dot(&self.parameters)
                                - self.learned_gravity_projection;
                        gravity_square_sum += residual * residual;
                        gravity_count += 1;
                    }
                }
                (gravity_count > 0).then_some(gravity_square_sum / gravity_count as f32)
            } else {
                None
            };
        let coverage = self.mean_centered_coverage();
        let radial_fitness = Self::radial_fitness_score(Some(radial_mean_square));
        let gravity_fitness = Self::gravity_fitness_score(gravity_mean_square);
        self.quality = CalibrationQuality::new(coverage, radial_fitness, gravity_fitness);
        Some(candidate)
    }
}
