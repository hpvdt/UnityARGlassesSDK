use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use super::bad_mag_cause::BadCalibration;
use super::mag_calibrator::{ONLINE_SCALE_EPSILON, SHAPE_PRIOR_SCALE};
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
/// Radial algebraic-residual RMS at which the radial fitness reaches 0,
/// ramping linearly from 1 at RMS 0. The residual lives in normalized
/// cache coordinates, so the constant is calibrated against the fixed-seed
/// SimMotion regression, where converged fits score algebraic RMS roughly
/// `0.06`–`0.15`, and against the Air 1 replay, where the post-warmup
/// average stays near `0.11`. The former physical corrected-radius
/// residual (`MAX_RADIAL_RMS` `0.1`) lived on a different scale —
/// algebraically the two differ by roughly `2 * gamma` plus quadratic
/// outlier weighting — so the old constant does not transfer.
pub(super) const MAX_RADIAL_RMS: f32 = 0.5;
/// Gravity-projection RMS residual below which the gravity fitness is 1.
/// The residual is relative to the projection scale $\sigma_g$, so the floor
/// is a dip-inconsistency fraction: at 0.1 the dip projection of the
/// retained rows may spread by a tenth of its own mean magnitude before the
/// fitness leaves the plateau. A perfect fit under clean hints keeps a
/// residual far below this; the floor absorbs the transient spread while the
/// preconditioner frame is still converging — and under its clamp bound —
/// so that transient bias never drags down a good calibration.
const GRAVITY_RMS_FLOOR: f32 = 0.1;
/// Eigenvalue bounds of the gravity preconditioner `gravity_frame`
/// ($A_w^{-1}$). While the working correction converges, preconditioning is a
/// fixed-point iteration: each refresh retargets the surrogate at the current
/// working anisotropy and the optimizer then moves the correction. Clamping
/// bounds the anisotropy the iteration can inject per refresh, keeping the
/// moving target stable. Only anisotropy matters to the surrogate — a common
/// scale factor of the frame is absorbed by the learned projection $\kappa$ —
/// so a well-converged frame sits comfortably inside these bounds and the
/// clamp engages only while the working shape is still far off.
const MIN_GRAVITY_FRAME_EIGENVALUE: f32 = 0.25;
/// Upper eigenvalue bound of `gravity_frame`; see `MIN_GRAVITY_FRAME_EIGENVALUE`.
const MAX_GRAVITY_FRAME_EIGENVALUE: f32 = 4.0;
/// Gravity-projection RMS residual at which the gravity fitness reaches 0,
/// ramping linearly down from 1 at `GRAVITY_RMS_FLOOR`. The residual is
/// relative to the projection scale $\sigma_g$ (a dip-inconsistency
/// fraction), so the constant transfers across devices and field radii; it
/// is calibrated against the synthetic consistent/contradictory gravity
/// tests (a fully contradictory hint stream sits near the relative RMS of
/// 1) and against the Air 1 trace, whose steady accelerometer-hint spread
/// centers near 0.2.
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
    /// sample normalization $u_i = (x_i - \mu) / r$. Zero on an empty or
    /// single-point cache, which makes the normalization unusable; see
    /// `sample_normalization_usable`.
    pub(super) sample_rms_radius: f32,
    /// Learned scalar $\kappa$ of the gravity surrogate: the projection of
    /// the preconditioned gravity direction $\tilde{g}_i = A_w^{-1} g_i$
    /// onto the ellipsoid normal $n_i = Q u_i + q / 2$ at a retained row,
    /// $\kappa = \psi(u_i, \tilde{g}_i)^T \theta$, which the surrogate keeps
    /// approximately constant across rows. `None` until seeded once from the
    /// first usable gravity observation, then refined by the optimizer
    /// gradient steps.
    pub(super) learned_gravity_projection: Option<f32>,
    /// Gravity preconditioner frame $A_w^{-1}$: the inverse of the current
    /// working soft-iron correction, symmetrized with eigenvalues clamped to
    /// [`MIN_GRAVITY_FRAME_EIGENVALUE`, `MAX_GRAVITY_FRAME_EIGENVALUE`]. The
    /// surrogate residual uses the preconditioned direction
    /// $\tilde{g}_i = A_w^{-1} g_i$, so it pins
    /// $\tilde{g}_i^T n_i = \gamma r\, g_i^T A_w^{-1} A m_i$, which reduces
    /// to the exact magnetic dip $\gamma r\, g_i^T m_i$ once the working
    /// correction $A_w$ matches the true $A$. Identity until the first valid
    /// working candidate refreshes it; refreshed by [`MagModel::update_quality`].
    pub(super) gravity_frame: Matrix3<f32>,
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

    /// Whether the sample normalization $(\mu, r)$ of the retained
    /// magnetometer samples is usable: both finite and the radius above
    /// `f32::EPSILON`, which requires two distinct samples. Derived from the
    /// normalization fields at point of use, so usability can never disagree
    /// with the state it describes.
    pub(super) fn sample_normalization_usable(&self) -> bool {
        self.sample_mean.iter().all(|value| value.is_finite())
            && self.sample_rms_radius.is_finite()
            && self.sample_rms_radius > f32::EPSILON
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
    /// makes the normalization unusable, which keeps the optimizer
    /// idle until two distinct samples exist and reports quality zero
    /// through the usual unusable-candidate path.
    pub(super) fn refresh_normalization(&mut self) {
        let Some((sample_mean, covariance)) = self.raw_mean_and_covariance() else {
            self.sample_mean = Vector3::zeros();
            self.sample_rms_radius = 0.0;
            return;
        };
        self.sample_mean = sample_mean;
        self.sample_rms_radius = covariance.trace().sqrt();
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
    /// otherwise the first retained row that does. The seed uses the same
    /// preconditioned features as the optimizer and the quality statistic,
    /// $\kappa = \psi(u, \tilde{g})^T \theta$.
    pub(super) fn initialize_gravity_projection(
        &mut self,
        current_sample: Vector3<f32>,
        current_gravity: Option<Vector3<f32>>,
    ) {
        if self.learned_gravity_projection.is_some() || self.gravity_weight == 0.0 {
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
            let features = Self::gravity_features(
                self.normalized_sample(sample),
                self.preconditioned_gravity(gravity),
            );
            let projection = features.dot(&self.parameters);
            if projection.is_finite() {
                self.learned_gravity_projection = Some(projection);
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
    /// parameters, keeping the combined online objective convex and
    /// quadratic. `gravity` is the preconditioned direction $\tilde{g}$ —
    /// callers apply [`MagModel::preconditioned_gravity`] first.
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

    /// Preconditioned gravity direction $\tilde{g} = A_w^{-1} g$ used by the
    /// surrogate everywhere (seeding, optimizer features, live quality).
    /// Deliberately not renormalized: the projection residual must equal
    /// $g^T A_w^{-1} n$, and a common scale factor of the frame is absorbed
    /// by the learned projection $\kappa$.
    pub(super) fn preconditioned_gravity(&self, gravity: Vector3<f32>) -> Vector3<f32> {
        self.gravity_frame * gravity
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
    /// `MAX_RADIAL_RMS`, applied to the mean square of the algebraic
    /// ellipsoid residual `phi(u_i)^T theta - 1` recomputed over the
    /// retained rows with the current working parameters. This is exactly
    /// the radial data term `e_{r,i}` the online optimizer minimizes, so a
    /// fitness drop directly signals optimization regress rather than a
    /// mismatch between two differently scaled residuals. A missing or
    /// unusable statistic scores 0.
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
    /// recomputed over the retained rows carrying a gravity direction, with
    /// `psi` built from the preconditioned direction $\tilde{g}_i$. Unlike
    /// the radial score, a missing statistic maps to a neutral 1: gravity
    /// is optional, so an absent or disabled gravity term must never
    /// penalize a magnetometer-only calibration. Preconditioning removes
    /// the surrogate's anisotropic soft-iron bias as the frame converges;
    /// the remaining transient bias stays inside the floor.
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
        if !self.sample_normalization_usable() {
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

    /// Refreshes the gravity preconditioner frame from a valid working
    /// candidate: $A_w^{-1}$ is the inverse of the candidate's correction,
    /// symmetrized, with eigenvalues clamped to
    /// [`MIN_GRAVITY_FRAME_EIGENVALUE`, `MAX_GRAVITY_FRAME_EIGENVALUE`]. The
    /// candidate correction is already SPD with bounded condition, so the
    /// inverse is well-defined; the clamp bounds the per-refresh target
    /// motion of the fixed-point iteration between preconditioner and fit.
    /// The frame is refreshed even when the surrogate is weight-disabled, so
    /// a later opt-in never starts from a stale frame.
    pub(super) fn refresh_gravity_frame(&mut self, candidate: &CalibrationCandidate) {
        let Some(inverse) = candidate.correction.try_inverse() else {
            return;
        };
        let symmetrized = 0.5 * (inverse + inverse.transpose());
        let eigen = symmetrized.symmetric_eigen();
        let clamped = eigen
            .eigenvalues
            .map(|value| value.clamp(MIN_GRAVITY_FRAME_EIGENVALUE, MAX_GRAVITY_FRAME_EIGENVALUE));
        self.gravity_frame =
            eigen.eigenvectors * Matrix3::from_diagonal(&clamped) * eigen.eigenvectors.transpose();
    }

    /// Updates the live quality of the current working candidate.
    /// Normalization uses the maintained raw moments; coverage and both
    /// fitness statistics rescan the retained rows, so an expired or
    /// replaced row stops contributing to the reported quality immediately.
    ///
    /// Historical note: with the earlier physical residual
    /// `||A (x_i - b)|| - 1` the Air 1 replay showed block-long post-warmup
    /// radial-fitness dips to zero. The physical residual scales against the
    /// optimizer's algebraic residual by the state-dependent factor
    /// `2 * gamma` and warps outliers differently, so the statistic could
    /// degrade while the optimizer kept descending its own objective.
    /// Recomputing fitness from the algebraic residual
    /// `phi(u_i)^T theta - 1` — the optimizer's own data term — removed the
    /// dips: the Air 1 post-warmup fitness now stays above the 0.5
    /// stability floor for thousands of consecutive evaluations.
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
        self.refresh_gravity_frame(&candidate);
        // Radial fitness: mean square of the algebraic ellipsoid residual
        // `phi(u_i)^T theta - 1` over every retained row, in fixed ascending
        // row order so the result is bit-deterministic. This is the same
        // data term the online optimizer minimizes in normalized cache
        // coordinates, so the fitness directly tracks optimization progress
        // on the retained support; the regularization prior is deliberately
        // excluded, keeping the statistic a pure data-fit measure. A
        // non-finite accumulation marks the statistic unusable, matching
        // the zero-quality path above.
        let mut radial_square_sum = 0.0f32;
        for row in 0..self.sample_row_count {
            let residual = Self::features(self.normalized_sample(self.samples.view(row).sample()))
                .dot(&self.parameters)
                - 1.0;
            radial_square_sum += residual * residual;
        }
        let radial_mean_square = radial_square_sum / self.sample_row_count as f32;
        if !radial_mean_square.is_finite() {
            self.quality = CalibrationQuality::ZERO;
            return None;
        }
        // Gravity fitness: mean square of the projection residual over the
        // retained rows that carry a gravity direction, normalized by the
        // projection scale $\sigma_g$ (RMS projection over the same rows),
        // exactly as in the online objective: the relative dip-residual is
        // device-independent, while the raw residual scales with the field
        // radius $r$. The statistic stays absent (neutral 1 below) when
        // gravity is disabled, the projection is not yet seeded, or no
        // retained row carries gravity.
        let mut gravity_square_sum = 0.0f32;
        let mut gravity_scale_square_sum = 0.0f32;
        let mut gravity_count = 0usize;
        let gravity_mean_square = match self.learned_gravity_projection {
            Some(kappa) if self.gravity_weight > 0.0 => {
                for row in 0..self.sample_row_count {
                    let row = self.samples.view(row);
                    if let Some(gravity) = row.gravity() {
                        let projection = Self::gravity_features(
                            self.normalized_sample(row.sample()),
                            self.preconditioned_gravity(gravity),
                        )
                        .dot(&self.parameters);
                        let residual = projection - kappa;
                        gravity_square_sum += residual * residual;
                        gravity_scale_square_sum += projection * projection;
                        gravity_count += 1;
                    }
                }
                (gravity_count > 0).then(|| {
                    let scale_squared = (gravity_scale_square_sum / gravity_count as f32)
                        .max(ONLINE_SCALE_EPSILON);
                    gravity_square_sum / gravity_count as f32 / scale_squared
                })
            }
            _ => None,
        };
        let coverage = self.mean_centered_coverage();
        let radial_fitness = Self::radial_fitness_score(Some(radial_mean_square));
        let gravity_fitness = Self::gravity_fitness_score(gravity_mean_square);
        // The gravity factor joins the combined fitness with the same
        // relative weight the objective gives it: `w_g` while the statistic
        // is live, zero otherwise (a neutral gravity factor must not shift
        // the exponent off pure radial).
        let gravity_term_weight = if gravity_mean_square.is_some() {
            self.gravity_weight
        } else {
            0.0
        };
        self.quality = CalibrationQuality::new(
            coverage,
            radial_fitness,
            gravity_fitness,
            gravity_term_weight,
        );
        Some(candidate)
    }
}
