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
/// Eigenvalue bounds of the gravity preconditioner `gravity_frame`
/// ($A_w^{-1}$), as multiples of the frame's mean eigenvalue. While the
/// working correction converges, preconditioning is a fixed-point iteration:
/// each refresh retargets the surrogate at the current working anisotropy and
/// the optimizer then moves the correction. Clamping bounds the anisotropy
/// the iteration can inject per refresh, keeping the moving target stable.
/// Only anisotropy matters to the surrogate — a common scale factor of the
/// frame is absorbed by the learned projection $\kappa$ — and the frame's
/// eigenvalue scale tracks the raw sample radius $r$ (tens of microtesla on
/// real traces), so the bounds are relative to the mean eigenvalue rather
/// than absolute: absolute bounds would saturate every eigenvalue of a
/// real-scale frame at the ceiling and erase the anisotropy the
/// preconditioner exists to remove. A well-converged frame sits comfortably
/// inside these relative bounds and the clamp engages only while the working
/// shape is still far off.
const MIN_GRAVITY_FRAME_EIGENVALUE: f32 = 0.25;
/// Upper eigenvalue bound of `gravity_frame`; see `MIN_GRAVITY_FRAME_EIGENVALUE`.
const MAX_GRAVITY_FRAME_EIGENVALUE: f32 = 4.0;
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
    /// Learned dip projection $g^T m$ of this candidate,
    /// $\kappa / (\gamma r)$: the projection of the calibrated unit
    /// magnetic field onto the caller-provided gravity hint direction.
    /// `None` while the learned projection $\kappa$ is unseeded or gravity
    /// is disabled.
    pub(super) dip_sin: Option<f32>,
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
    /// TODO: several fields (e.g. stats, MagCalibrator.sample_timestamps_us) can be moved into `samples: MagSamples<N>`
    ///  MagSamples.set_row function should also keep these fields up-to-date
    ///
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
    /// working soft-iron correction, symmetrized with each eigenvalue clamped
    /// to [`MIN_GRAVITY_FRAME_EIGENVALUE`, `MAX_GRAVITY_FRAME_EIGENVALUE`]
    /// multiples of the frame's mean eigenvalue. The
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
    /// `update_quality` on every publication evaluation; its `dip_sin`
    /// record is retained across unusable candidates, mirroring the
    /// last-known-good fallback of the published correction.
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
    /// update (not per observation), and the live radial loss reports the
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

        // The surrogate keeps the normal projection
        // $\psi(u_i, \tilde{g}_i)^T \theta \approx \kappa$, which the
        // ellipsoid identity reduces to $\gamma r\, g_i^T m_i$ once the
        // preconditioner matches the true correction, so $\kappa/(\gamma r)$
        // is the dip projection $g^T m$ in the caller's hint convention.
        let dip_sin = if self.gravity_weight > 0.0 {
            self.learned_gravity_projection
                .map(|kappa| kappa / (ellipsoid_scale * r))
                .filter(|value| value.is_finite())
        } else {
            None
        };

        Ok(CalibrationCandidate {
            offset,
            correction,
            dip_sin,
        })
    }

    /// Refreshes the gravity preconditioner frame from a valid working
    /// candidate: $A_w^{-1}$ is the inverse of the candidate's correction,
    /// symmetrized, with each eigenvalue clamped to
    /// [`MIN_GRAVITY_FRAME_EIGENVALUE`, `MAX_GRAVITY_FRAME_EIGENVALUE`]
    /// multiples of the frame's mean eigenvalue. The candidate correction is
    /// already SPD with bounded condition, so the inverse is well-defined;
    /// the mean-relative clamp bounds the anisotropy the iteration can inject
    /// per refresh while preserving the frame's common scale, keeping the
    /// moving target of the fixed-point iteration between preconditioner and
    /// fit stable. The frame is refreshed even when the surrogate is
    /// weight-disabled, so a later opt-in never starts from a stale frame.
    pub(super) fn refresh_gravity_frame(&mut self, candidate: &CalibrationCandidate) {
        // Defensive: a valid working candidate is SPD with bounded condition,
        // so inversion cannot fail; keep the previous frame if it ever does.
        let Some(inverse) = candidate.correction.try_inverse() else {
            return;
        };
        let symmetrized = 0.5 * (inverse + inverse.transpose());
        let eigen = symmetrized.symmetric_eigen();
        // The frame's eigenvalue scale tracks the raw sample radius $r$, so
        // the anisotropy clamp is applied relative to the mean eigenvalue:
        // only anisotropy matters to the surrogate, and the common scale the
        // clamp preserves is absorbed by the learned projection $\kappa$.
        let mean_eigenvalue = eigen.eigenvalues.sum() / 3.0;
        if !mean_eigenvalue.is_finite() || mean_eigenvalue <= f32::EPSILON {
            return;
        }
        let clamped = eigen.eigenvalues.map(|value| {
            (value / mean_eigenvalue)
                .clamp(MIN_GRAVITY_FRAME_EIGENVALUE, MAX_GRAVITY_FRAME_EIGENVALUE)
                * mean_eigenvalue
        });
        self.gravity_frame =
            eigen.eigenvectors * Matrix3::from_diagonal(&clamped) * eigen.eigenvectors.transpose();
    }

    /// Updates the live quality of the current working candidate.
    /// Normalization uses the maintained raw moments; coverage and both
    /// loss statistics rescan the retained rows, so an expired or
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
    /// dips in the reported radial statistic. The physical residual scales
    /// against the optimizer's algebraic residual by the state-dependent
    /// factor `2 * gamma` and warps outliers differently, so the statistic
    /// could degrade while the optimizer kept descending its own objective.
    /// Recomputing the loss from the algebraic residual
    /// `phi(u_i)^T theta - 1` — the optimizer's own data term — removed the
    /// dips: the Air 1 post-warmup radial loss stayed low for thousands of
    /// consecutive evaluations.
    pub(super) fn update_quality(&mut self) -> Option<CalibrationCandidate> {
        // The dip projection record of the quality is refreshed only by a
        // usable candidate below and retained across unusable ones,
        // mirroring the last-known-good fallback of the published
        // correction.
        let retained_dip_sin = self.quality.dip_sin;
        let reset_quality = |quality: &mut CalibrationQuality| {
            *quality = CalibrationQuality {
                dip_sin: retained_dip_sin,
                ..CalibrationQuality::ZERO
            };
        };
        if self.stats.sample_row_count < CALIBRATION_PARAMETER_COUNT {
            reset_quality(&mut self.quality);
            return None;
        }
        let candidate = match self.working_candidate() {
            Ok(candidate) => candidate,
            Err(_) => {
                reset_quality(&mut self.quality);
                return None;
            }
        };
        self.refresh_gravity_frame(&candidate);
        let (mu /*$\mu$*/, rms_radius /*$r$*/) = self.stats.normalization();
        // The gravity loss stays absent (zero below) when gravity is
        // disabled, the projection $\kappa$ is not yet seeded, or no
        // retained row carries gravity.
        let gravity_kappa = if self.gravity_weight > 0.0 {
            self.learned_gravity_projection
        } else {
            None
        };
        // The radial loss accumulates the square of the algebraic
        // ellipsoid residual `phi(u_i)^T theta - 1` over every retained
        // row, in fixed ascending row order so the result is
        // bit-deterministic, and the shape-regularization loss below adds
        // once — exactly the radial online objective $J_r$ the optimizer
        // minimizes in normalized cache coordinates (the data term is
        // per-observation averaged while the regularizer enters once).
        //
        // The gravity loss accumulates the projection residual
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
        let mut gram_sum = [[0.0f32; CALIBRATION_PARAMETER_COUNT]; CALIBRATION_PARAMETER_COUNT];
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
                // Hand-rolled symmetric outer-product accumulation into a
                // plain array: adds `feature[i] * feature[j]` to every Gram
                // entry, the same element-wise operations the matrix
                // expression `gram_sum += feature * feature.transpose()`
                // performs, without materializing the intermediate 9x9
                // product or routing each element access through the
                // generic matrix `Index` machinery.
                let components = feature.as_slice();
                for i in 0..CALIBRATION_PARAMETER_COUNT {
                    let component = components[i];
                    for j in 0..CALIBRATION_PARAMETER_COUNT {
                        gram_sum[i][j] += component * components[j];
                    }
                }
            }
        }
        let gram_sum = CoverageGramMatrix::from_fn(|i, j| gram_sum[i][j]);
        let regularization_loss = Self::regularization_loss(&self.parameters);
        let radial_loss =
            0.5 * radial_square_sum / self.stats.sample_row_count as f32 + regularization_loss;
        if !radial_loss.is_finite() {
            reset_quality(&mut self.quality);
            return None;
        }
        let gravity_mean_square = (gravity_count > 0).then(|| {
            let scale_squared =
                (gravity_scale_square_sum / gravity_count as f32).max(ONLINE_SCALE_EPSILON);
            gravity_square_sum / gravity_count as f32 / scale_squared
        });
        // The gravity term of the online objective over the retained rows:
        // $J_g = \frac{w_g}{2 n_g} \sum_i e_{g,i}^2$. A missing or non-finite
        // statistic reports zero, like a disabled gravity term: the
        // objective then contains no gravity term to report.
        let gravity_loss = gravity_mean_square
            .map(|mean_square| 0.5 * self.gravity_weight * mean_square)
            .filter(|loss| loss.is_finite())
            .unwrap_or(0.0);
        let coverage = Self::coverage_from_gram(&gram_sum, self.stats.sample_row_count);
        self.quality = CalibrationQuality {
            coverage,
            radial_loss,
            regularization_loss,
            gravity_loss,
            dip_sin: candidate.dip_sin,
        };
        Some(candidate)
    }
}
