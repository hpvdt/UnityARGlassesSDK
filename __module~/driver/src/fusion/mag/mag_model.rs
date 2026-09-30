use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use super::bad_mag_cause::BadCalibration;
use super::mag_calibrator::ONLINE_SCALE_EPSILON;
use super::mag_samples::{MagSamples, Row};
use super::sample_stats::SampleStats;
use super::CalibrationQuality;

/// Number of ellipsoid coefficients fitted by the magnetometer calibration
/// model.
pub(super) const CALIBRATION_PARAMETER_COUNT: usize = 9;

/// Gram sum of the retained direction features,
/// $\sum_i \varphi(\hat{u}_i) \varphi(\hat{u}_i)^T$, backing the coverage
/// score.
pub(super) type CoverageGramMatrix =
    SMatrix<f32, CALIBRATION_PARAMETER_COUNT, CALIBRATION_PARAMETER_COUNT>;
const MAX_CORRECTION_CONDITION: f32 = 1.0e1;
/// RMS-equivalent radial objective $\sqrt{2 J_r}$ at which the radial
/// fitness reaches 0, ramping linearly from 1 at 0. $J_r$ is the full
/// radial online objective: the mean square algebraic residual scaled by
/// $1/2$ plus the shape regularizer, so the scored RMS-equivalent is
/// `sqrt(mean_square + lambda * ||Q - c * I||_F^2)`. The residual lives in
/// normalized cache coordinates, so the constant is calibrated against the
/// fixed-seed SimMotion regression, where converged fits score roughly
/// `0.06`–`0.15`, and against the Air 1 replay, where the post-warmup
/// average stays near `0.11`; the regularizer adds only a small offset at
/// convergence. The former physical corrected-radius residual
/// (`MAX_RADIAL_RMS` `0.1`) lived on a different scale — algebraically the
/// two differ by roughly `2 * gamma` plus quadratic outlier weighting — so
/// the old constant does not transfer.
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
/// tests (a fully contradictory hint stream has a relative RMS near 1) and
/// against the Air 1 trace, whose steady accelerometer-hint spread centers
/// near 0.2.
const MAX_GRAVITY_RMS: f32 = 0.35;
/// Weight of the shape regularizer in the radial online objective
/// $J_r = \frac{1}{2 n} \sum_i e_{r,i}^2 + \frac{\lambda}{2} \|Q - c I\|_F^2$.
pub(super) const SHAPE_REGULARIZATION /*$\lambda$*/: f32 = 1.0e-3;
/// Scale of the regularization target shape, in units of the identity.
/// Algebraic ellipsoid fits under noise systematically inflate the ellipsoid
/// (underestimate the eigenvalues of the shape matrix), so the prior centers
/// on a shape larger than the ideal sphere to counter that bias.
pub(super) const SHAPE_PRIOR_SCALE /*$c$*/: f32 = 2.0;
/// Uniform-sphere reference for directional coverage: the smallest
/// eigenvalue of $\mathbb{E}[\varphi(\hat{u}) \varphi(\hat{u})^T]$ over
/// uniformly distributed unit directions $\hat{u}$, where $\varphi$ is the
/// coverage-feature vector with $\sqrt{2}$ cross-term weights (see
/// `coverage_feature`). A fully isotropic cache scores 1 against this
/// reference.
const COVERAGE_LAMBDA_REF: f32 = 2.0 / 15.0;

/// Finite hard-iron offset and soft-iron correction derived from the
/// working ellipsoid state of a [`MagModel`], ready to be published by the
/// calibrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CalibrationCandidate {
    pub(super) offset: Vector3<f32>,     /*$b$*/
    pub(super) correction: Matrix3<f32>, /*$A$*/
}

/// Calibration model state behind `MagCalibrator`: the retained magnetometer
/// sample cache the quality statistics are estimated from, the online
/// ellipsoid coefficients with the incrementally maintained cache statistics
/// ([`SampleStats`]) giving the sample normalization they are expressed in,
/// the learned gravity-projection state of the optional surrogate, and the
/// live quality factors derived from all of the above. Grouping the fields
/// keeps the quality-estimation inputs (`update_quality`) together and
/// separate from the optimizer bookkeeping, diversity neighbor cache, and
/// publication state that the calibrator owns itself.
pub(super) struct MagModel<const N: usize> {
    /// Retained magnetometer sample cache: the raw samples and the optional
    /// gravity direction carried by each row.
    pub(super) samples: MagSamples<N>,
    pub(super) parameters: SVector<f32, CALIBRATION_PARAMETER_COUNT>, /*$\theta$*/
    /// Incrementally maintained statistics of the retained cache rows:
    /// the row count and raw moments backing the sample normalization
    /// $(\mu, r)$ it derives. Grouped in [`SampleStats`] so append,
    /// replacement, and expiry update all of them together in $O(1)$
    /// without a row scan.
    /// TODO: this can be moved into `samples: MagSamples<N>`
    pub(super) stats: SampleStats,
    /// Learned scalar $\kappa$ of the gravity surrogate: the projection of
    /// the preconditioned gravity direction $\tilde{g}_i = A_w^{-1} g_i$
    /// onto the ellipsoid normal $n_i = Q u_i + q / 2$ at a retained row,
    /// $\kappa = \psi(u_i, \tilde{g}_i)^T \theta$, which the surrogate keeps
    /// approximately constant across rows. `None` until seeded once from the
    /// first usable gravity observation, then refined by the optimizer
    /// gradient steps.
    ///
    // TODO: extract gravity_frame & learned_gravity_projection into a new optional struct called "LearnedGravityState"
    //  which contains 2 fields, both are none-optional
    //  the instance here will be optional, indicating that LearnedState are either available or not
    pub(super) learned_gravity_projection: Option<f32>, /*$\kappa$*/
    /// Gravity preconditioner frame $A_w^{-1}$: the inverse of the current
    /// working soft-iron correction, symmetrized with eigenvalues clamped to
    /// [`MIN_GRAVITY_FRAME_EIGENVALUE`, `MAX_GRAVITY_FRAME_EIGENVALUE`]. The
    /// surrogate residual uses the preconditioned direction
    /// $\tilde{g}_i = A_w^{-1} g_i$, so it pins
    /// $\tilde{g}_i^T n_i = \gamma r\, g_i^T A_w^{-1} A m_i$, which reduces
    /// to the exact magnetic dip $\gamma r\, g_i^T m_i$ once the working
    /// correction $A_w$ matches the true $A$. Identity until the first valid
    /// working candidate refreshes it; refreshed by [`MagModel::update_quality`].
    pub(super) gravity_frame: Matrix3<f32>, /*$A_w^{-1}$*/
    pub(super) gravity_weight: f32, /*$w_g$*/
    /// Live calibration quality factors of the current working candidate,
    /// reset together with the model minimum and recomputed by
    /// `update_quality` on every publication evaluation.
    pub(super) quality: CalibrationQuality,
}

impl<const N: usize> MagModel<N> {
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

    /// Shape-regularization loss of the packed coefficients:
    /// $\frac{\lambda}{2} \|Q - c I\|_F^2$. This is the regularization term
    /// of the radial online objective $J_r$; the optimizer adds it once per
    /// update (not per observation), and the live radial fitness scores the
    /// same combined objective, so the term is shared here.
    pub(super) fn regularization_loss(
        parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    ) -> f32 {
        let (shape, _) = Self::unpack_ellipsoid_coefficients(parameters);
        0.5 * SHAPE_REGULARIZATION
            * (shape - Matrix3::identity() * SHAPE_PRIOR_SCALE).norm_squared()
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
                (0..self.stats.sample_row_count).find_map(|row| {
                    let row = self.samples.view(row);
                    row.gravity().map(|gravity| (row.sample(), gravity))
                })
            });
        if let Some((sample, gravity)) = observation {
            let features = Self::gravity_features(
                self.stats.normalized_sample(sample),
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

    /// Feature vector $\varphi(\hat{u})$ of a unit direction $\hat{u}$: the
    /// term whose outer products $\varphi(\hat{u}) \varphi(\hat{u})^T$ build
    /// the coverage Gram matrix summed by `update_quality` and scored by
    /// `coverage_from_gram`. The nine components are the ellipsoid-fit
    /// features with $\sqrt{2}$ cross-term weights; with that weighting the
    /// feature norm equals the rotation-invariant
    /// $\operatorname{tr}(\hat{u} \hat{u}^T \hat{u} \hat{u}^T)$, so the
    /// induced rotation on feature space is orthogonal and the Gram
    /// eigenvalues are exactly rotation-invariant. Under the uniform
    /// spherical distribution $\mathbb{E}[\varphi \varphi^T]$ has
    /// eigenvalues $\{1/3 \times 4,\ 2/15 \times 5\}$; the smallest,
    /// $2/15$, is the `COVERAGE_LAMBDA_REF` uniform-sphere reference.
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

    /// Radial fitness in `[0, 1]`: a linear ramp from 1 at zero RMS to 0 at
    /// `MAX_RADIAL_RMS`, applied to the mean square of the algebraic
    /// ellipsoid residual `phi(u_i)^T theta - 1` plus twice the
    /// shape-regularization loss — i.e. the RMS-equivalent `sqrt(2 J_r)` of
    /// the full radial objective — recomputed over the retained rows with
    /// the current working parameters. This is exactly the radial loss
    /// $J_r$ the online optimizer minimizes, so a fitness drop directly
    /// signals optimization regress rather than a mismatch between two
    /// differently scaled residuals. A missing or unusable statistic scores
    /// 0.
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
        if !self.stats.sample_normalization_usable() {
            return Err(BadCalibration::Unsolveable {
                message: "sample normalization is non-finite or zero",
            });
        }
        let (mu /*$\mu$*/, r /*$r$*/) = self.stats.normalization();
        let parameters = self.parameters;
        if !parameters.iter().all(|value| value.is_finite()) {
            return Err(BadCalibration::Unsolveable {
                message: "online calibration produced non-finite parameters",
            });
        }

        let (shape /*$Q$*/, linear /*$q$*/) = Self::unpack_ellipsoid_coefficients(&parameters);
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
        let normalized_offset /*$d$*/ = -0.5 * shape_cholesky.solve(&linear);
        let ellipsoid_scale /*$\gamma$*/ = 1.0 + normalized_offset.dot(&(shape * normalized_offset));
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
            shape_eigen.eigenvectors * square_root * shape_eigen.eigenvectors.transpose() / r;
        let offset = mu + r * normalized_offset;
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
        // Defensive: a valid working candidate is SPD with bounded condition,
        // so inversion cannot fail; keep the previous frame if it ever does.
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
    /// All three row statistics share one pass over the retained rows: the
    /// normalization $(\mu, r)$ is derived once from the maintained raw
    /// moments, each row is read and centered exactly once, and the pass
    /// accumulates the radial residual sum, the gravity residual sums, and
    /// the coverage Gram sum together. The merge is bit-equivalent to the
    /// former three separate scans: every accumulator still sums its
    /// per-row terms in ascending row order, and the shared normalization
    /// and centering evaluate the same expressions the separate scans
    /// evaluated per row.
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
        if self.stats.sample_row_count < CALIBRATION_PARAMETER_COUNT {
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
        let (mu /*$\mu$*/, rms_radius /*$r$*/) = self.stats.normalization();
        // The gravity statistic stays absent (neutral 1 below) when gravity
        // is disabled, the projection $\kappa$ is not yet seeded, or no
        // retained row carries gravity.
        let gravity_kappa = if self.gravity_weight > 0.0 {
            self.learned_gravity_projection
        } else {
            None
        };
        // Radial fitness accumulates the mean square of the algebraic
        // ellipsoid residual `phi(u_i)^T theta - 1` over every retained
        // row, in fixed ascending row order so the result is
        // bit-deterministic, plus twice the shape-regularization loss
        // below — exactly the radial online objective $J_r$ the optimizer
        // minimizes in normalized cache coordinates (the data term is
        // per-observation averaged while the regularizer enters once), so
        // the fitness ramp below scores the same loss the optimizer
        // descends and its square root is the RMS-equivalent
        // $\sqrt{2 J_r}$ of the combined objective.
        //
        // Gravity fitness accumulates the projection residual
        // `psi(u_i, g_i)^T theta - kappa` and its scale sum over the
        // retained rows carrying a gravity direction, normalized by the
        // projection scale $\sigma_g$ (RMS projection over the same rows,
        // floored below), exactly as in the online objective: the relative
        // dip-residual is device-independent, while the raw residual
        // scales with the field radius $r$.
        //
        // Coverage accumulates the Gram sum of the mean-centered unit
        // directions; the cache mean is the center rather than the fitted
        // hard-iron offset because the offset's component along the
        // thinnest data direction is itself unconstrained for near-planar
        // support, which destabilizes the score exactly where it must be
        // decisive.
        let mut radial_square_sum = 0.0f32;
        let mut gravity_square_sum = 0.0f32;
        let mut gravity_scale_square_sum = 0.0f32;
        let mut gravity_count = 0usize;
        let mut gram_sum = CoverageGramMatrix::zeros();
        for row in 0..self.stats.sample_row_count {
            let row_view = self.samples.view(row);
            let centered = row_view.sample() - mu;
            let normalized = centered / rms_radius;
            let residual = Self::features(normalized).dot(&self.parameters) - 1.0;
            radial_square_sum += residual * residual;
            if let Some(kappa) = gravity_kappa {
                if let Some(gravity) = row_view.gravity() {
                    let projection =
                        Self::gravity_features(normalized, self.preconditioned_gravity(gravity))
                            .dot(&self.parameters);
                    let residual = projection - kappa;
                    gravity_square_sum += residual * residual;
                    gravity_scale_square_sum += projection * projection;
                    gravity_count += 1;
                }
            }
            if let Some(direction) = centered.try_normalize(f32::EPSILON) {
                let feature = Self::coverage_feature(direction);
                gram_sum += feature * feature.transpose();
            }
        }
        let regularization_loss = Self::regularization_loss(&self.parameters);
        let radial_objective_mean_square =
            radial_square_sum / self.stats.sample_row_count as f32 + 2.0 * regularization_loss;
        if !radial_objective_mean_square.is_finite() {
            self.quality = CalibrationQuality::ZERO;
            return None;
        }
        let gravity_mean_square = (gravity_count > 0).then(|| {
            let scale_squared =
                (gravity_scale_square_sum / gravity_count as f32).max(ONLINE_SCALE_EPSILON);
            gravity_square_sum / gravity_count as f32 / scale_squared
        });
        let coverage = Self::coverage_from_gram(&gram_sum, self.stats.sample_row_count);
        let radial_fitness = Self::radial_fitness_score(Some(radial_objective_mean_square));
        let gravity_fitness = Self::gravity_fitness_score(gravity_mean_square);
        self.quality = CalibrationQuality::new(
            coverage,
            radial_fitness,
            regularization_loss,
            gravity_fitness,
        );
        Some(candidate)
    }
}
