use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use super::bad_mag_cause::{BadCalibration, BadMagCause, BadReading};

const CALIBRATION_PARAMETER_COUNT: usize = 9;
/// Gram sum of the retained direction features, `sum_i phi(d_i) phi(d_i)^T`,
/// backing the coverage score.
pub(super) type CoverageGramMatrix =
    SMatrix<f32, CALIBRATION_PARAMETER_COUNT, CALIBRATION_PARAMETER_COUNT>;
const SHAPE_REGULARIZATION: f32 = 1.0e-3;
/// Scale of the regularization target shape, in units of the identity.
/// Algebraic ellipsoid fits under noise systematically inflate the ellipsoid
/// (underestimate the eigenvalues of the shape matrix), so the prior centers
/// on a shape larger than the ideal sphere to counter that bias.
const SHAPE_PRIOR_SCALE: f32 = 2.0;
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
/// against the synthetic consistent/contradictory gravity test (steady-state
/// RMS ~0.14 consistent, ~0.25 contradictory) pending benchmark validation.
const MAX_GRAVITY_RMS: f32 = 0.3;
/// Confidence required for a working candidate to advance the publication
/// streak. This is the highest tested threshold at which every fixed SimMotion
/// regression seed completes the 2000-evaluation budget; the rank-deficient
/// planar regression remains below it.
pub(super) const MIN_PUBLICATION_CONFIDENCE: f32 = 0.0125;
/// Confidence floor below which the publication streak resets. This
/// hysteresis keeps a qualifying candidate from losing its streak to
/// threshold jitter: confidence in `[0.01, 0.0125)` pauses the streak while a
/// drop below `0.01` resets it.
const PUBLICATION_STREAK_RESET_CONFIDENCE: f32 = 0.01;
/// Valid updates a working candidate must hold at least
/// `MIN_PUBLICATION_CONFIDENCE` before publishing. Halved from 110 so the
/// SimMotion integration benchmark completes within its 2000-evaluation
/// budget; the hysteresis floor still rejects jitter, and 55 valid updates
/// is about 1.1 s of sustained quality at the 50 Hz magnetometer rate.
pub(super) const MIN_PUBLICATION_STREAK: usize = 55;
/// Uniform-sphere reference for directional coverage: the smallest
/// eigenvalue of `E[phi(d) phi(d)^T]` over uniformly distributed unit
/// directions, where `phi` is the quadratic feature vector shared with the
/// ellipsoid fit. A fully isotropic cache scores 1 against this reference.
const COVERAGE_LAMBDA_REF: f32 = 2.0 / 15.0;
const MIN_MAG_NORM: f32 = 0.4;
/// Default weight of the ellipsoid-normal gravity surrogate. The surrogate
/// pins `g_i^T A m_i` (with `A` the soft-iron correction) approximately
/// constant instead of the exact magnetic dip `g_i^T m_i`; the two coincide
/// only for isotropic soft iron, so strong anisotropic soft iron can bias the
/// fit toward isotropy. The fixed-seed benchmark found that weight `0.1`
/// regressed accuracy, while lowering the default to `0.01` recovered average
/// post-warm-up accuracy to within `0.086 degree` of the direct gravity
/// baseline. Validation must keep extending beyond the fixed simulator
/// distortion (stronger anisotropy, rotated eigenvectors, inconsistent
/// acceleration, multiple magnetic dip angles); lower or disable the surrogate
/// through [`MagCalibrator::gravity_weight`] if such sweeps show a repeatable
/// regression.
const DEFAULT_GRAVITY_WEIGHT: f32 = 0.01;
const DEFAULT_MINIBATCH_SIZE: usize = 32;
/// Cache-only replay updates run per valid sample while the calibration is
/// still unpublished. They let the cold-start optimizer take several gradient
/// steps per arriving sample without waiting for new data.
const DEFAULT_REPLAY_UPDATES: usize = 4;
/// Replay minibatches are smaller than the sample-anchored minibatch because
/// several of them run per sample and each one re-evaluates its objective
/// during the bounded half-step search.
const DEFAULT_REPLAY_MINIBATCH_SIZE: usize = 8;
const ONLINE_INITIAL_LEARNING_RATE: f32 = 0.5;
/// Learning-rate annealing timescale, in optimizer steps. The target ellipsoid
/// is not stationary: it keeps moving as long as cache replacements improve
/// the sample coverage, which under near-planar motion continues well past the
/// `N`-sample cache fill. The rate must therefore stay high enough through and
/// beyond the fill for the optimizer to track the moving convex optimum. A
/// timescale far below the fill time collapses the rate before convergence and
/// strands the working shape far from the optimum (seen as >18 deg worst-case
/// SimMotion-integration error with a near-planar seed at 64 steps).
const ONLINE_LEARNING_RATE_DECAY_STEPS: f32 = 128.0;
const ONLINE_MIN_LEARNING_RATE: f32 = 0.01;
const ONLINE_MAX_STEP_NORM: f32 = 0.5;
const ONLINE_SCALE_EPSILON: f32 = 1.0e-4;
const ONLINE_BACKTRACK_STEPS: usize = 12;
const ONLINE_PRNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Number of neighbor entries cached per buffered sample row: the `k` nearest
/// squared distances plus an overshoot pad that absorbs neighbor churn before
/// an O(N) row rescan becomes necessary. Configurations with `num_neighbors`
/// above this capacity bypass the cache and scan rows directly.
const NEIGHBOR_CACHE_CAPACITY: usize = 8;

/// Squared distance from one buffered sample row to another, addressed by row
/// index.
#[derive(Clone, Copy)]
struct NeighborEntry {
    squared_distance: f32,
    row: u32,
}

impl NeighborEntry {
    /// Placeholder for cache slots past the row's trusted prefix length; such
    /// slots are never read.
    const EMPTY: Self = Self {
        squared_distance: f32::INFINITY,
        row: 0,
    };
}

#[derive(Clone, Copy)]
struct MinibatchSpec {
    /// The arriving observation anchoring a sample-triggered update.
    /// Cache-replay updates leave this empty and draw every observation from
    /// the retained rows.
    current_sample: Option<Vector3<f32>>,
    current_gravity: Option<Vector3<f32>>,
    accepted_row: Option<usize>,
    random_draws: usize,
    random_state: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CalibrationCandidate {
    offset: Vector3<f32>,
    correction: Matrix3<f32>,
}

/// Live calibration quality factors of the current working candidate, all
/// bounded in `[0, 1]`. Kept as one unit because they are stored as state,
/// reset together, and reported together as the quality half of
/// [`MagCalibrationResult`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationQuality {
    /// Current bounded calibration quality in `[0, 1]`: the clamped product
    /// of `coverage` and `fitness`.
    pub confidence: f32,
    /// Directional coverage factor of the confidence in `[0, 1]`: the
    /// E-optimality score of the retained mean-centered unit directions.
    pub coverage: f32,
    /// Combined fitness factor of the confidence in `[0, 1]`:
    /// `radial_fitness * gravity_fitness`.
    pub fitness: f32,
    /// Radial fitness sub-factor in `[0, 1]`: the bounded fit of the
    /// working correction over recent valid samples.
    pub radial_fitness: f32,
    /// Gravity-consistency fitness sub-factor in `[0, 1]`: the bounded
    /// running fit of the gravity-projection surrogate. `1.0` while no
    /// valid gravity direction has been seen or the gravity term is
    /// disabled, so a magnetometer-only stream is never penalized.
    pub gravity_fitness: f32,
}

impl CalibrationQuality {
    const ZERO: Self = Self {
        confidence: 0.0,
        coverage: 0.0,
        fitness: 0.0,
        radial_fitness: 0.0,
        gravity_fitness: 0.0,
    };

    fn new(coverage: f32, radial_fitness: f32, gravity_fitness: f32) -> Self {
        let fitness = radial_fitness * gravity_fitness;
        let quality = coverage * fitness;
        Self {
            confidence: if quality.is_finite() {
                quality.clamp(0.0, 1.0)
            } else {
                0.0
            },
            coverage,
            fitness,
            radial_fitness,
            gravity_fitness,
        }
    }
}

/// Result of evaluating one FRD magnetometer observation.
///
/// The quality factors are exposed directly through [`Deref`] to
/// [`CalibrationQuality`], so `result.confidence` reads the same as
/// `result.quality.confidence`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MagCalibrationResult {
    /// Live calibration quality factors of the current working candidate.
    pub quality: CalibrationQuality,
    /// Corrected and normalized FRD magnetic direction, produced by the
    /// published correction; `None` while no correction has passed the live
    /// quality gates yet.
    pub direction: Option<Vector3<f32>>,
}

impl std::ops::Deref for MagCalibrationResult {
    type Target = CalibrationQuality;

    fn deref(&self) -> &Self::Target {
        &self.quality
    }
}

impl MagCalibrationResult {
    fn from_quality(quality: CalibrationQuality, direction: Option<Vector3<f32>>) -> Self {
        Self { quality, direction }
    }
}

/// Online regularized ellipsoid fit for a hard-iron offset and full SPD
/// soft-iron correction from a fixed, diverse sample buffer.
pub struct MagCalibrator<const N: usize> {
    sample_matrix: SMatrix<f32, N, 3>,
    gravity_directions: [Option<Vector3<f32>>; N],
    sample_timestamps_us: [u64; N],
    sample_row_count: usize,
    hard_iron_offset: Vector3<f32>,
    soft_iron_correction: Matrix3<f32>,
    calibration_initialized: bool,
    mean_distance: f32,
    /// Per-row incremental k-nearest-neighbor cache for the diversity
    /// heuristic. Invariant: `neighbor_cache[i][..neighbor_cache_len[i]]`
    /// lists, in ascending squared distance, the true nearest other buffered
    /// rows of row `i` (the row itself is excluded by index), and every
    /// buffered row not listed is at least as far as the last listed entry.
    /// The trusted prefix is updated in place on append and replace, remapped
    /// on expiry compaction, and rebuilt with an O(N) scan once it shrinks
    /// below `k`.
    neighbor_cache: [[NeighborEntry; NEIGHBOR_CACHE_CAPACITY]; N],
    neighbor_cache_len: [u8; N],
    neighbor_count: usize,
    max_sample_lifespan_us: u64,
    gravity_weight: f32,
    parameters: SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    normalization_mean: Vector3<f32>,
    normalization_radius: f32,
    normalization_initialized: bool,
    gravity_projection: f32,
    gravity_projection_initialized: bool,
    minibatch_size: usize,
    replay_updates: usize,
    replay_minibatch_size: usize,
    prng_state: u64,
    optimizer_steps: u64,
    raw_sample_sum: Vector3<f64>,
    raw_outer_product_sum: Matrix3<f64>,
    radial_residual_mean_square: Option<f32>,
    gravity_residual_mean_square: Option<f32>,
    quality: CalibrationQuality,
    publication_quality_streak: usize,
}

impl<const N: usize> Default for MagCalibrator<N> {
    fn default() -> Self {
        Self {
            sample_matrix: SMatrix::zeros(),
            gravity_directions: std::array::from_fn(|_| None),
            sample_timestamps_us: [0; N],
            sample_row_count: Default::default(),
            hard_iron_offset: Vector3::zeros(),
            soft_iron_correction: Matrix3::identity(),
            calibration_initialized: false,
            mean_distance: Default::default(),
            neighbor_cache: [[NeighborEntry::EMPTY; NEIGHBOR_CACHE_CAPACITY]; N],
            neighbor_cache_len: [0; N],
            neighbor_count: 2, // Works well in testing
            max_sample_lifespan_us: 60 * 60 * 1_000_000,
            gravity_weight: DEFAULT_GRAVITY_WEIGHT,
            parameters: Self::parameter_prior(),
            normalization_mean: Vector3::zeros(),
            normalization_radius: 0.0,
            normalization_initialized: false,
            gravity_projection: 0.0,
            gravity_projection_initialized: false,
            minibatch_size: DEFAULT_MINIBATCH_SIZE.min(N.max(1)),
            replay_updates: DEFAULT_REPLAY_UPDATES,
            replay_minibatch_size: DEFAULT_REPLAY_MINIBATCH_SIZE.min(N.max(1)),
            prng_state: ONLINE_PRNG_SEED,
            optimizer_steps: 0,
            raw_sample_sum: Vector3::zeros(),
            raw_outer_product_sum: Matrix3::zeros(),
            radial_residual_mean_square: None,
            gravity_residual_mean_square: None,
            quality: CalibrationQuality::ZERO,
            publication_quality_streak: 0,
        }
    }
}

impl<const N: usize> MagCalibrator<N> {
    /// Create a new calibrator instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the number of `k` neighbors to calculate distance to.
    pub fn num_neighbors(self, neighbor_count: usize) -> Self {
        Self {
            neighbor_count: neighbor_count.clamp(1, N.saturating_sub(1).max(1)),
            ..self
        }
    }

    /// Configure the maximum time a sample remains in the calibration buffer,
    /// in microseconds. The default is one hour.
    ///
    /// Known adaptation limitation (see `AGENTS.md`): expiring or replacing a
    /// row removes it from the cache but not its historical online-SGD
    /// gradient contribution, so this bounds cache membership rather than the
    /// optimizer's effective history. The backlog is an explicit forgetting
    /// schedule or bounded replay-after-expiry, plus an adaptive hard-iron
    /// drift case.
    pub fn max_sample_lifespan_us(self, max_sample_lifespan_us: u64) -> Self {
        Self {
            max_sample_lifespan_us,
            ..self
        }
    }

    /// Configure the relative weight of the gravity-consistency residual.
    /// The default is 0.01; zero disables the ellipsoid-normal gravity
    /// surrogate.
    pub fn gravity_weight(self, gravity_weight: f32) -> Self {
        Self {
            gravity_weight: if gravity_weight.is_finite() {
                gravity_weight.max(0.0)
            } else {
                0.0
            },
            ..self
        }
    }

    /// Configure the maximum number of observations used by each online
    /// optimizer update. The current valid observation is always included. The
    /// default is 32, capped by the sample-buffer capacity.
    pub fn minibatch_size(self, minibatch_size: usize) -> Self {
        Self {
            minibatch_size: minibatch_size.clamp(1, N.max(1)),
            ..self
        }
    }

    /// Configure the number of additional cache-only optimizer updates run
    /// with each valid sample while the calibration is still unpublished.
    /// Replay draws its whole minibatch from the retained rows, so it never
    /// requires the arriving sample; it accelerates cold-start convergence at
    /// a small pre-publication computation cost. The default is 4; zero
    /// disables replay.
    pub fn replay_updates(self, replay_updates: usize) -> Self {
        Self {
            replay_updates,
            ..self
        }
    }

    /// Configure the number of retained-row observations used by each
    /// cache-replay update. The default is 8, smaller than the
    /// sample-anchored minibatch, capped by the sample-buffer capacity.
    pub fn replay_minibatch_size(self, replay_minibatch_size: usize) -> Self {
        Self {
            replay_minibatch_size: replay_minibatch_size.clamp(1, N.max(1)),
            ..self
        }
    }

    fn parameter_prior() -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
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

    fn shape_and_linear(
        parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>,
    ) -> (Matrix3<f32>, Vector3<f32>) {
        // Diagonal [Q00, Q11, Q22] followed by packed off-diagonal [Q01, Q02, Q12].
        let shape = Matrix3::from_fn(|row, col| {
            let index = if row == col { row } else { row + col + 2 };
            parameters[index]
        });
        (shape, parameters.fixed_rows::<3>(6).into_owned())
    }

    fn features(sample: Vector3<f32>) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
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

    /// Features of the projection of gravity onto the ellipsoid normal
    /// `Q * sample + q / 2`. The projection is linear in the nine ellipsoid
    /// parameters, keeping the combined online objective convex and quadratic.
    fn gravity_features(
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

    fn regularization_loss(parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>) -> f32 {
        let (shape, _) = Self::shape_and_linear(parameters);
        0.5 * SHAPE_REGULARIZATION
            * (shape - Matrix3::identity() * SHAPE_PRIOR_SCALE).norm_squared()
    }

    fn add_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum += sample;
        self.raw_outer_product_sum += sample * sample.transpose();
    }

    fn remove_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum -= sample;
        self.raw_outer_product_sum -= sample * sample.transpose();
    }

    fn clear_raw_moments(&mut self) {
        self.raw_sample_sum = Vector3::zeros();
        self.raw_outer_product_sum = Matrix3::zeros();
    }

    fn raw_mean_and_covariance(&self) -> Option<(Vector3<f32>, Matrix3<f32>)> {
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
    /// expiry drifts the mean and radius; the working coefficients keep
    /// their meaning in the new normalization directly, because the drift
    /// per cache mutation is `O(1 / sample_row_count)` and the online optimizer
    /// is already designed to track the moving convex optimum as cache
    /// replacements improve coverage. Working state is therefore never
    /// rebased or reset: only a zero-radius (empty or single-point) cache
    /// marks the normalization uninitialized, which keeps the optimizer
    /// idle until two distinct samples exist and reports quality zero
    /// through the usual unusable-candidate path.
    fn refresh_normalization(&mut self) {
        let Some((sample_mean, covariance)) = self.raw_mean_and_covariance() else {
            self.normalization_mean = Vector3::zeros();
            self.normalization_radius = 0.0;
            self.normalization_initialized = false;
            return;
        };
        let radius = covariance.trace().sqrt();
        self.normalization_initialized = sample_mean.iter().all(|value| value.is_finite())
            && radius.is_finite()
            && radius > f32::EPSILON;
        self.normalization_mean = sample_mean;
        self.normalization_radius = radius;
    }

    fn normalized_sample(&self, sample: Vector3<f32>) -> Vector3<f32> {
        (sample - self.normalization_mean) / self.normalization_radius
    }

    fn next_random(random_state: &mut u64) -> u64 {
        *random_state = random_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *random_state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn random_cache_row(
        random_state: &mut u64,
        sample_count: usize,
        excluded_row: Option<usize>,
    ) -> Option<usize> {
        let eligible_count = sample_count.saturating_sub(usize::from(excluded_row.is_some()));
        if eligible_count == 0 {
            return None;
        }
        let mut row = (Self::next_random(random_state) % eligible_count as u64) as usize;
        if excluded_row.is_some_and(|excluded| row >= excluded) {
            row += 1;
        }
        Some(row)
    }

    fn initialize_gravity_projection(
        &mut self,
        current_sample: Vector3<f32>,
        current_gravity: Option<Vector3<f32>>,
    ) {
        if self.gravity_projection_initialized || self.gravity_weight == 0.0 {
            return;
        }
        let observation = current_gravity
            .map(|gravity| (current_sample, gravity))
            .or_else(|| {
                (0..self.sample_row_count).find_map(|row| {
                    self.gravity_directions[row].map(|gravity| (self.sample(row), gravity))
                })
            });
        if let Some((sample, gravity)) = observation {
            let features = Self::gravity_features(self.normalized_sample(sample), gravity);
            let projection = features.dot(&self.parameters);
            if projection.is_finite() {
                self.gravity_projection = projection;
                self.gravity_projection_initialized = true;
            }
        }
    }

    /// Stacks feature rows into a batched `B x 9` matrix.
    fn feature_matrix(rows: &[SVector<f32, CALIBRATION_PARAMETER_COUNT>]) -> DMatrix<f32> {
        DMatrix::from_fn(rows.len(), CALIBRATION_PARAMETER_COUNT, |row, col| {
            rows[row][col]
        })
    }

    /// Reinterprets the nine-entry dynamic result of a batched feature-matrix
    /// product as a fixed parameter vector.
    fn parameter_vector(vector: DVector<f32>) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
        SVector::from_column_slice(vector.as_slice())
    }

    /// Chains the anchoring observation with the randomly drawn cache rows of
    /// one minibatch and returns their batched radial and gravity feature
    /// matrices along with the advanced private draw state. Gravity rows cover
    /// only the observations carrying a usable gravity direction.
    fn minibatch_feature_matrices(
        &self,
        minibatch: MinibatchSpec,
    ) -> (DMatrix<f32>, DMatrix<f32>, u64) {
        let mut random_state = minibatch.random_state;
        let observations = minibatch
            .current_sample
            .map(|sample| (sample, minibatch.current_gravity))
            .into_iter()
            .chain((0..minibatch.random_draws).map_while(|_| {
                Self::random_cache_row(
                    &mut random_state,
                    self.sample_row_count,
                    minibatch.accepted_row,
                )
                .map(|row| (self.sample(row), self.gravity_directions[row]))
            }));
        let mut radial_rows = Vec::with_capacity(minibatch.random_draws + 1);
        let mut gravity_rows = Vec::with_capacity(minibatch.random_draws + 1);
        for (sample, gravity) in observations {
            let normalized = self.normalized_sample(sample);
            radial_rows.push(Self::features(normalized));
            if let Some(gravity) = gravity.filter(|_| self.gravity_projection_initialized) {
                gravity_rows.push(Self::gravity_features(normalized, gravity));
            }
        }
        (
            Self::feature_matrix(&radial_rows),
            Self::feature_matrix(&gravity_rows),
            random_state,
        )
    }

    fn minibatch_objective(
        &self,
        parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>,
        gravity_projection: f32,
        minibatch: MinibatchSpec,
    ) -> f32 {
        let (features, gravity_features, _) = self.minibatch_feature_matrices(minibatch);
        let residuals = &features * parameters - DVector::from_element(features.nrows(), 1.0);
        let mut objective = 0.5 * residuals.norm_squared() / features.nrows() as f32
            + Self::regularization_loss(parameters);
        if gravity_features.nrows() > 0 {
            let residuals = &gravity_features * parameters
                - DVector::from_element(gravity_features.nrows(), gravity_projection);
            objective += 0.5 * self.gravity_weight * residuals.norm_squared()
                / gravity_features.nrows() as f32;
        }
        objective
    }

    /// Applies one bounded normalized-SGD update anchored to the current
    /// valid sample, followed, while the calibration is still unpublished, by
    /// the configured number of cache-replay updates drawn purely from the
    /// retained rows.
    fn update_online_optimizer(
        &mut self,
        current_sample: Vector3<f32>,
        current_gravity: Option<Vector3<f32>>,
        accepted_row: Option<usize>,
    ) {
        if !self.normalization_initialized {
            return;
        }
        self.initialize_gravity_projection(current_sample, current_gravity);

        let random_draws = if self.sample_row_count > usize::from(accepted_row.is_some()) {
            self.minibatch_size.saturating_sub(1)
        } else {
            0
        };
        if self.apply_minibatch_update(MinibatchSpec {
            current_sample: Some(current_sample),
            current_gravity,
            accepted_row,
            random_draws,
            random_state: self.prng_state,
        }) {
            self.optimizer_steps += 1;
        }

        // Cold-start cache replay: the arriving sample is never a required
        // member of a replay minibatch; once retained, it is an ordinary
        // cache row that replay may draw like any other. Replay ramps in with
        // the retained fraction: repeatedly fitting a small, low-coverage
        // cache overfits it and can strand the working shape outside the
        // publishable region, while a nearly full cache is representative
        // enough to converge against. Replay steps reuse the current
        // learning rate without advancing its schedule, so annealing stays
        // tied to the rate of arriving data rather than to compute.
        if self.calibration_initialized || self.sample_row_count == 0 {
            return;
        }
        let replay_count = self.replay_updates.saturating_mul(self.sample_row_count) / N.max(1);
        for _ in 0..replay_count {
            self.apply_minibatch_update(MinibatchSpec {
                current_sample: None,
                current_gravity: None,
                accepted_row: None,
                random_draws: self.replay_minibatch_size,
                random_state: self.prng_state,
            });
        }
    }

    /// Applies one bounded normalized-SGD update over the given minibatch,
    /// advancing the private draw state past the sampled rows. Returns whether
    /// a finite objective-lowering step was accepted; an update that finds no
    /// observation or no usable descent direction leaves the working state
    /// unchanged.
    fn apply_minibatch_update(&mut self, minibatch: MinibatchSpec) -> bool {
        let (features, gravity_features, random_state) = self.minibatch_feature_matrices(minibatch);
        self.prng_state = random_state;
        if features.nrows() == 0 {
            return false;
        }

        let parameters = self.parameters;
        let gravity_projection = self.gravity_projection;
        let residuals = &features * parameters - DVector::from_element(features.nrows(), 1.0);
        let mut gradient =
            Self::parameter_vector(features.tr_mul(&residuals)) / features.nrows() as f32;
        let mut gradient_scale =
            Self::parameter_vector(features.map(|value| value * value).row_sum_tr())
                / features.nrows() as f32;
        let mut gravity_projection_gradient = 0.0;
        if gravity_features.nrows() > 0 {
            let residuals = &gravity_features * parameters
                - DVector::from_element(gravity_features.nrows(), gravity_projection);
            gradient += self.gravity_weight
                * Self::parameter_vector(gravity_features.tr_mul(&residuals))
                / gravity_features.nrows() as f32;
            gradient_scale += self.gravity_weight
                * Self::parameter_vector(gravity_features.map(|value| value * value).row_sum_tr())
                / gravity_features.nrows() as f32;
            gravity_projection_gradient =
                -residuals.sum() * (self.gravity_weight / gravity_features.nrows() as f32);
        }

        let prior = Self::parameter_prior();
        let regularization_weights =
            SVector::<f32, CALIBRATION_PARAMETER_COUNT>::from_row_slice(&[
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
            ]);
        gradient +=
            SHAPE_REGULARIZATION * regularization_weights.component_mul(&(parameters - prior));
        gradient_scale += SHAPE_REGULARIZATION * regularization_weights;
        gradient_scale.add_scalar_mut(ONLINE_SCALE_EPSILON);
        let descent_direction = gradient.component_div(&gradient_scale);
        let projection_step = if gravity_features.nrows() > 0 && self.gravity_weight > 0.0 {
            gravity_projection_gradient / (self.gravity_weight + ONLINE_SCALE_EPSILON)
        } else {
            0.0
        };
        // TODO: use built-in norm operations instead of manually combining vector and scalar squared norms
        let descent_norm =
            (descent_direction.norm_squared() + projection_step * projection_step).sqrt();
        if !descent_norm.is_finite() || descent_norm <= f32::EPSILON {
            return false;
        }

        let old_objective = self.minibatch_objective(&parameters, gravity_projection, minibatch);
        let learning_rate = (ONLINE_INITIAL_LEARNING_RATE
            / (1.0 + self.optimizer_steps as f32 / ONLINE_LEARNING_RATE_DECAY_STEPS))
            .max(ONLINE_MIN_LEARNING_RATE);
        let mut step_size = learning_rate.min(ONLINE_MAX_STEP_NORM / descent_norm);
        for _ in 0..ONLINE_BACKTRACK_STEPS {
            let trial_parameters = parameters - step_size * descent_direction;
            let trial_gravity_projection = gravity_projection - step_size * projection_step;
            let objective =
                self.minibatch_objective(&trial_parameters, trial_gravity_projection, minibatch);
            if trial_parameters.iter().all(|value| value.is_finite())
                && trial_gravity_projection.is_finite()
                && objective.is_finite()
                && objective < old_objective
            {
                self.parameters = trial_parameters;
                self.gravity_projection = trial_gravity_projection;
                return true;
            }
            step_size *= 0.5;
        }
        false
    }

    /// Computes squared distances from `mag_sample` to the first `count`
    /// rows of the sample buffer. Entries at and beyond `count` are set to
    /// infinity so selection never picks them.
    fn squared_distances_to(&self, mag_sample: Vector3<f32>, count: usize) -> [f32; N] {
        let mut squared_distances = [f32::INFINITY; N];
        for (j, dist) in squared_distances.iter_mut().enumerate().take(count) {
            *dist = (mag_sample - self.sample(j)).norm_squared();
        }
        squared_distances
    }

    /// Mean distance over the `k` smallest entries of `squared_distances`,
    /// selected in O(n) with a partial sort; `squared_distances` is reordered
    /// in the process. The square root is deferred until after selection, so
    /// only the `k` selected entries are sqrt'd. Returns infinity for
    /// `k == 0`.
    fn mean_of_smallest(squared_distances: &mut [f32], neighbor_count: usize) -> f32 {
        if neighbor_count == 0 {
            return f32::INFINITY;
        }
        squared_distances.select_nth_unstable_by(neighbor_count - 1, |a, b| a.total_cmp(b));
        let smallest = &mut squared_distances[..neighbor_count];
        smallest.sort_unstable_by(|a, b| a.total_cmp(b));
        smallest.iter().map(|&d| d.sqrt()).sum::<f32>() / neighbor_count as f32
    }

    /// Inserts `entry` into a row's neighbor cache, keeping it sorted and
    /// bounded by the capacity. An entry ranking beyond the trusted prefix is
    /// only appended when the cache currently covers every other buffered row
    /// (`complete`); otherwise it is dropped, because an uncached row may
    /// legitimately be closer and would silently break the prefix invariant.
    fn cache_insert(
        cache: &mut [NeighborEntry; NEIGHBOR_CACHE_CAPACITY],
        len: &mut u8,
        entry: NeighborEntry,
        complete: bool,
    ) {
        let count = *len as usize;
        let rank = cache[..count].partition_point(|e| e.squared_distance < entry.squared_distance);
        if rank < count {
            let shift_end = count.min(NEIGHBOR_CACHE_CAPACITY - 1);
            cache.copy_within(rank..shift_end, rank + 1);
            cache[rank] = entry;
            if count < NEIGHBOR_CACHE_CAPACITY {
                *len += 1;
            }
        } else if complete && count < NEIGHBOR_CACHE_CAPACITY {
            cache[count] = entry;
            *len += 1;
        }
    }

    /// Removes the entry referencing `row` from a neighbor cache, if present.
    /// Dropping an entry keeps the remaining prefix trusted.
    fn cache_remove(cache: &mut [NeighborEntry; NEIGHBOR_CACHE_CAPACITY], len: &mut u8, row: u32) {
        let count = *len as usize;
        if let Some(position) = cache[..count].iter().position(|e| e.row == row) {
            cache.copy_within(position + 1..count, position);
            *len -= 1;
        }
    }

    /// Rebuilds a row's neighbor cache from scratch: the
    /// `NEIGHBOR_CACHE_CAPACITY` smallest squared distances among rows
    /// `0..count`, skipping the row's own entry by index. O(N).
    fn reset_row_cache(&mut self, row: usize, squared_distances: &[f32; N], count: usize) {
        let mut entries = [NeighborEntry::EMPTY; N];
        let mut entry_count = 0;
        for (j, &squared_distance) in squared_distances.iter().enumerate().take(count) {
            if j == row {
                continue;
            }
            entries[entry_count] = NeighborEntry {
                squared_distance,
                row: j as u32,
            };
            entry_count += 1;
        }
        let take = NEIGHBOR_CACHE_CAPACITY.min(entry_count);
        if take > 0 {
            entries[..entry_count].select_nth_unstable_by(take - 1, |a, b| {
                a.squared_distance.total_cmp(&b.squared_distance)
            });
            entries[..take]
                .sort_unstable_by(|a, b| a.squared_distance.total_cmp(&b.squared_distance));
        }
        self.neighbor_cache[row][..take].copy_from_slice(&entries[..take]);
        self.neighbor_cache_len[row] = take as u8;
    }

    /// Recomputes a row's neighbor cache when its trusted prefix has shrunk
    /// below `k`. Only called with a full buffer.
    fn rebuild_row_cache(&mut self, row: usize) {
        let squared_distances = self.squared_distances_to(self.sample(row), N);
        self.reset_row_cache(row, &squared_distances, N);
    }

    /// Mean distance of a buffered row to its `k` nearest other rows, served
    /// from the incremental neighbor cache. A smaller number means the point
    /// is "similar" to its neighbors. Rows whose trusted prefix has shrunk
    /// below `k` are rescanned in O(N) first; configurations with `k` above
    /// the cache capacity always scan directly.
    fn row_mean_distance(&mut self, row: usize, neighbor_count: usize) -> f32 {
        if neighbor_count == 0 {
            return f32::INFINITY;
        }
        if neighbor_count > NEIGHBOR_CACHE_CAPACITY {
            return self.mean_distance_uncached(row, neighbor_count);
        }
        if (self.neighbor_cache_len[row] as usize) < neighbor_count {
            self.rebuild_row_cache(row);
        }
        let cache = &self.neighbor_cache[row];
        cache[..neighbor_count]
            .iter()
            .map(|entry| entry.squared_distance.sqrt())
            .sum::<f32>()
            / neighbor_count as f32
    }

    /// Direct O(N) computation of a row's mean distance to its `k` nearest
    /// other rows, used when `k` exceeds the neighbor cache capacity.
    fn mean_distance_uncached(&self, row: usize, neighbor_count: usize) -> f32 {
        let mut squared_distances = self.squared_distances_to(self.sample(row), N);
        // Skip the self-entry by index instead of dropping the smallest value.
        squared_distances[row] = f32::INFINITY;
        Self::mean_of_smallest(&mut squared_distances, neighbor_count)
    }

    /// Remaps cached neighbor row indices through `index_map` (`u32::MAX` =
    /// expired) after expiry compaction, shrinking trusted prefixes that
    /// referenced expired rows. Slots at and beyond the retained count keep
    /// stale values; they are reset on append before they can be read again.
    fn remap_neighbor_cache(&mut self, index_map: &[u32; N]) {
        for old_index in 0..N {
            let new_index = index_map[old_index];
            if new_index == u32::MAX {
                continue;
            }
            let mut cache = self.neighbor_cache[old_index];
            let count = self.neighbor_cache_len[old_index] as usize;
            let mut retained_count = 0;
            for i in 0..count {
                let entry = cache[i];
                let mapped = index_map[entry.row as usize];
                if mapped != u32::MAX {
                    cache[retained_count] = NeighborEntry {
                        squared_distance: entry.squared_distance,
                        row: mapped,
                    };
                    retained_count += 1;
                }
            }
            self.neighbor_cache[new_index as usize] = cache;
            self.neighbor_cache_len[new_index as usize] = retained_count as u8;
        }
    }

    /// Returns index of the buffered row with the lowest mean distance to its
    /// `k` nearest neighbors, derived from the incremental neighbor cache.
    /// Is used when replacing the least useful value in the array.
    fn lowest_mean_distance_by_index(&mut self) -> (usize, f32) {
        let neighbor_count = self.neighbor_count.min(N.saturating_sub(1));
        let mean_dist =
            SVector::<f32, N>::from_fn(|index, _| self.row_mean_distance(index, neighbor_count));
        self.mean_distance = mean_dist.mean();
        mean_dist.argmin()
    }

    /// Add a sample if it is deemed more useful than the least useful sample.
    ///
    /// `gravity_direction` is an optional co-timestamped body-frame FRD
    /// direction. Non-finite and zero directions are ignored. The live quality
    /// and publication state are updated even when diversity rejects the valid
    /// current observation.
    pub fn evaluate_sample_vec(
        &mut self,
        mag_sample: Vector3<f32>,
        gravity_hint: Option<Vector3<f32>>,
        timestamp_us: u64,
    ) {
        let gravity_direction = gravity_hint.and_then(|direction| {
            direction
                .try_normalize(f32::EPSILON)
                .filter(|direction| direction.iter().all(|value| value.is_finite()))
        });
        let valid_current_sample = self.ingest_sample(mag_sample, gravity_direction, timestamp_us);
        self.update_publication(
            valid_current_sample.then_some(mag_sample),
            gravity_direction,
        );
    }

    /// Updates the cache and online optimizer, returning whether the current
    /// magnetometer observation was finite and nonzero. `gravity_direction`
    /// must already be normalized to a unit vector.
    fn ingest_sample(
        &mut self,
        mag_sample: Vector3<f32>,
        gravity_direction: Option<Vector3<f32>>,
        timestamp_us: u64,
    ) -> bool {
        let previous_sample_row_count = self.sample_row_count;
        let mut index_map = [u32::MAX; N];
        let mut retained_count = 0;
        for (index, map_slot) in index_map.iter_mut().enumerate().take(self.sample_row_count) {
            let sample = self.sample(index);
            if timestamp_us.saturating_sub(self.sample_timestamps_us[index])
                <= self.max_sample_lifespan_us
            {
                *map_slot = retained_count as u32;
                if retained_count != index {
                    self.sample_matrix
                        .set_row(retained_count, &sample.transpose());
                    self.gravity_directions[retained_count] = self.gravity_directions[index];
                    self.sample_timestamps_us[retained_count] = self.sample_timestamps_us[index];
                }
                retained_count += 1;
            } else {
                self.remove_raw_moment(sample);
            }
        }
        if retained_count != self.sample_row_count {
            self.sample_row_count = retained_count;
            if retained_count == 0 {
                // Incremental subtraction can leave round-off residue after
                // the last retained row expires. An empty cache has exact
                // zero moments by definition.
                self.clear_raw_moments();
            }
            self.mean_distance = 0.0;
            self.remap_neighbor_cache(&index_map);
        }
        let expired = retained_count != previous_sample_row_count;

        if !mag_sample.iter().all(|e| e.is_finite()) || mag_sample.norm_squared() <= f32::EPSILON {
            if expired {
                self.refresh_normalization();
            }
            return false;
        }
        if N == 0 {
            return false;
        }
        let mut accepted_row = None;
        // Check if buffer is not yet "initialized" with real measurements
        if self.sample_row_count < N {
            let count = self.sample_row_count;
            let squared_distances = self.squared_distances_to(mag_sample, count);
            for ((cache, len), &squared_distance) in self
                .neighbor_cache
                .iter_mut()
                .zip(self.neighbor_cache_len.iter_mut())
                .zip(squared_distances.iter())
                .take(count)
            {
                // The cache covers every other row only if it was built up
                // without ever hitting the capacity or losing entries.
                let complete = *len as usize == count - 1;
                Self::cache_insert(
                    cache,
                    len,
                    NeighborEntry {
                        squared_distance,
                        row: count as u32,
                    },
                    complete,
                );
            }
            self.add_raw_moment(mag_sample);
            self.add_sample_at(count, mag_sample, gravity_direction, timestamp_us);
            self.reset_row_cache(count, &squared_distances, count);
            self.sample_row_count += 1;
            accepted_row = Some(count);
        }
        // Otherwise check which sample may be best to replace
        else {
            let neighbor_count = self.neighbor_count.min(N.saturating_sub(1));
            let (replacement_row, replacement_mean_distance) = self.lowest_mean_distance_by_index();
            let squared_distances = self.squared_distances_to(mag_sample, N);
            // The candidate has no self-entry in the buffer, so its mean
            // distance covers the true k nearest buffered rows.
            //
            // Known asymmetry (see TODO.md "Score replacement candidates in
            // their post-replacement buffer"): the victim's diversity score
            // (`replacement_mean_distance`) is its mean distance to its `k`
            // nearest OTHER rows, because `replacement_row` excludes itself,
            // so its pool is `N - 1` rows. The candidate below is instead
            // scored against all `N` old rows, including the victim row it
            // would replace, so its pool is `N` rows. A candidate close to
            // the victim can therefore be wrongly rejected because the
            // soon-to-be-evicted row lowers its nearest-neighbor score.
            //
            // Proposed fix (not implemented here): set
            // `candidate_squared_distances[replacement_row] = f32::INFINITY`
            // before selecting the `k` nearest neighbors so both scores use
            // the same `N - 1` retained rows, and update the incremental
            // neighbor cache only after accepting the replacement.
            let mut candidate_squared_distances = squared_distances;
            let candidate_mean_distance =
                Self::mean_of_smallest(&mut candidate_squared_distances, neighbor_count);
            if replacement_mean_distance < candidate_mean_distance {
                for (row, ((cache, len), &squared_distance)) in self
                    .neighbor_cache
                    .iter_mut()
                    .zip(self.neighbor_cache_len.iter_mut())
                    .zip(squared_distances.iter())
                    .enumerate()
                {
                    if row == replacement_row {
                        continue;
                    }
                    Self::cache_remove(cache, len, replacement_row as u32);
                    // After removal the cache covers every row besides the
                    // row itself and the replaced one only if nothing was
                    // ever evicted from it.
                    let complete = *len as usize == N.saturating_sub(2);
                    Self::cache_insert(
                        cache,
                        len,
                        NeighborEntry {
                            squared_distance,
                            row: replacement_row as u32,
                        },
                        complete,
                    );
                }
                self.remove_raw_moment(self.sample(replacement_row));
                self.add_raw_moment(mag_sample);
                self.add_sample_at(replacement_row, mag_sample, gravity_direction, timestamp_us);
                self.reset_row_cache(replacement_row, &squared_distances, N);
                accepted_row = Some(replacement_row);
            }
        }
        if expired || accepted_row.is_some() {
            self.refresh_normalization();
        }
        self.update_online_optimizer(mag_sample, gravity_direction, accepted_row);
        true
    }

    /// Insert a sample vector into the `index` row of the sample matrix.
    fn add_sample_at(
        &mut self,
        index: usize,
        sample: Vector3<f32>,
        gravity_direction: Option<Vector3<f32>>,
        timestamp_us: u64,
    ) {
        if index < N {
            self.sample_matrix.set_row(index, &sample.transpose());
            self.gravity_directions[index] = gravity_direction;
            self.sample_timestamps_us[index] = timestamp_us;
        }
    }

    /// Quadratic feature vector of a unit direction: the nine ellipsoid-fit
    /// features with `sqrt(2)` cross-term weights. With this weighting the
    /// feature norm equals the rotation-invariant `tr(d d^T d d^T)`, so the
    /// induced rotation on feature space is orthogonal and the Gram
    /// eigenvalues are exactly rotation-invariant. Under the uniform
    /// spherical distribution `E[phi phi^T]` has eigenvalues `{1/3 x4, 2/15
    /// x5}`.
    fn direction_feature(direction: Vector3<f32>) -> SVector<f32, CALIBRATION_PARAMETER_COUNT> {
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
    fn coverage_from_gram(gram_sum: &CoverageGramMatrix, sample_row_count: usize) -> f32 {
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
    fn mean_centered_coverage(&self) -> f32 {
        let mut gram_sum = CoverageGramMatrix::zeros();
        for row in 0..self.sample_row_count {
            let centered = self.sample(row) - self.normalization_mean;
            if let Some(direction) = centered.try_normalize(f32::EPSILON) {
                let feature = Self::direction_feature(direction);
                gram_sum += feature * feature.transpose();
            }
        }
        Self::coverage_from_gram(&gram_sum, self.sample_row_count)
    }

    /// Get mean distance value between samples in the sample matrix.
    pub fn get_mean_distance(&self) -> f32 {
        self.mean_distance
    }

    /// Returns the current bounded calibration quality in `[0, 1]`.
    ///
    /// Zero means the current working candidate is pending or unusable. A
    /// previously published correction can remain available while this value
    /// is zero after a rejected later candidate.
    pub fn get_confidence(&self) -> f32 {
        self.quality.confidence
    }

    /// Calibrates a magnetometer vector that has already been converted to FRD.
    ///
    /// `gravity_direction` is an optional co-timestamped body-frame FRD
    /// direction. It contributes a convex constant-projection surrogate to the
    /// online ellipsoid fit without making gravity mandatory for calibration.
    pub fn evaluate_correct(
        &mut self,
        raw_mag: Vector3<f32>,
        gravity_hint: Option<Vector3<f32>>,
        timestamp_us: u64,
    ) -> Result<MagCalibrationResult, BadMagCause> {
        self.evaluate_sample_vec(raw_mag, gravity_hint, timestamp_us);
        if !self.calibration_initialized {
            return Ok(MagCalibrationResult::from_quality(self.quality, None));
        }
        let mut mag = self.soft_iron_correction * (raw_mag - self.hard_iron_offset);

        let mag_norm = mag.normalize_mut();
        if !mag_norm.is_finite() || mag_norm < MIN_MAG_NORM {
            Err(BadMagCause::BadReading(BadReading::WeakCalibratedReading {
                norm: mag_norm,
                min_norm: MIN_MAG_NORM,
            }))
        } else {
            Ok(MagCalibrationResult::from_quality(self.quality, Some(mag)))
        }
    }

    fn update_publication(
        &mut self,
        current_sample: Option<Vector3<f32>>,
        current_gravity: Option<Vector3<f32>>,
    ) {
        let current_sample_valid = current_sample.is_some();
        let candidate = self.update_quality(current_sample, current_gravity);
        if !current_sample_valid || candidate.is_none() {
            // Invalid observations and unusable candidates always reset the
            // streak: they are evidence against publishing, not jitter.
            self.publication_quality_streak = 0;
        } else if self.quality.confidence >= MIN_PUBLICATION_CONFIDENCE {
            self.publication_quality_streak = self.publication_quality_streak.saturating_add(1);
        } else if self.quality.confidence < PUBLICATION_STREAK_RESET_CONFIDENCE {
            // Only a genuine quality collapse restarts the streak; a short
            // dip in live quality while the optimizer absorbs newly visited
            // directions merely pauses it.
            self.publication_quality_streak = 0;
        }
        if self.publication_quality_streak >= MIN_PUBLICATION_STREAK {
            if let Some(candidate) = candidate {
                self.hard_iron_offset = candidate.offset;
                self.soft_iron_correction = candidate.correction;
                self.calibration_initialized = true;
            }
        }
    }

    /// Derives one finite SPD correction candidate from the current online
    /// ellipsoid state without scanning retained rows.
    fn working_candidate(&self) -> Result<CalibrationCandidate, BadCalibration> {
        if !self.normalization_initialized
            || !self
                .normalization_mean
                .iter()
                .all(|value| value.is_finite())
            || !self.normalization_radius.is_finite()
            || self.normalization_radius <= f32::EPSILON
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

        let (shape, linear) = Self::shape_and_linear(&parameters);
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
                / self.normalization_radius;
        let offset = self.normalization_mean + self.normalization_radius * normalized_offset;
        if !offset.iter().all(|value| value.is_finite())
            || !correction.iter().all(|value| value.is_finite())
        {
            return Err(BadCalibration::Unsolveable {
                message: "calibration produced non-finite parameters",
            });
        }

        Ok(CalibrationCandidate { offset, correction })
    }

    fn update_running_mean_square(
        current: Option<f32>,
        residual_squared: f32,
        update_weight: f32,
    ) -> Option<f32> {
        if !residual_squared.is_finite()
            || residual_squared < 0.0
            || !update_weight.is_finite()
            || !(0.0..=1.0).contains(&update_weight)
            || update_weight == 0.0
        {
            return None;
        }
        let current = current.unwrap_or(0.0);
        if !current.is_finite() || current < 0.0 {
            return None;
        }
        let updated = current + update_weight * (residual_squared - current);
        updated.is_finite().then_some(updated.max(0.0))
    }

    fn radial_fitness_score(mean_square: Option<f32>) -> f32 {
        match mean_square {
            Some(mean_square) if mean_square.is_finite() && mean_square >= 0.0 => {
                (1.0 - mean_square.sqrt() / MAX_RADIAL_RMS).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    /// Gravity fitness in `[0, 1]`: a linear ramp from 1 at the
    /// `GRAVITY_RMS_FLOOR` residual to 0 at `MAX_GRAVITY_RMS`, applied to
    /// the running mean square of the gravity-projection residual
    /// `psi^T theta - kappa`. Unlike the radial score, a missing
    /// statistic maps to a neutral 1: gravity is optional, so an absent or
    /// disabled gravity term must never penalize a magnetometer-only
    /// calibration. The residual measures constancy of the ellipsoid-normal
    /// projection, which matches the corrected-direction dot product only
    /// for isotropic correction; the score inherits the surrogate's
    /// anisotropic soft-iron bias.
    fn gravity_fitness_score(mean_square: Option<f32>) -> f32 {
        match mean_square {
            Some(mean_square) if mean_square.is_finite() && mean_square >= 0.0 => {
                let rms = mean_square.sqrt();
                ((MAX_GRAVITY_RMS - rms) / (MAX_GRAVITY_RMS - GRAVITY_RMS_FLOOR)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        }
    }

    /// Updates the live radial and gravity statistics and quality for the
    /// current working candidate. All cache-dependent data comes from
    /// maintained moments.
    ///
    /// FIXME: the Air 1 replay shows block-long post-warm-up radial-fitness
    /// dips to zero even though the running statistic persists (it is never
    /// wiped) and grows smoothly through each dip. The dips are genuine
    /// working-candidate degradation on certain trace segments, not a
    /// normalization-lifecycle reset artifact, and persist unchanged across
    /// the removal of the rebase/reset path. Investigate why the online
    /// candidate degrades there instead of converging.
    fn update_quality(
        &mut self,
        current_sample: Option<Vector3<f32>>,
        current_gravity: Option<Vector3<f32>>,
    ) -> Option<CalibrationCandidate> {
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
        if let Some(sample) = current_sample {
            let residual = (candidate.correction * (sample - candidate.offset)).norm() - 1.0;
            let residual_squared = residual * residual;
            let update_weight = 1.0 / self.sample_row_count.min(self.minibatch_size).max(1) as f32;
            let Some(mean_square) = Self::update_running_mean_square(
                self.radial_residual_mean_square,
                residual_squared,
                update_weight,
            ) else {
                self.radial_residual_mean_square = None;
                self.quality = CalibrationQuality::ZERO;
                return None;
            };
            self.radial_residual_mean_square = Some(mean_square);
            if let Some(gravity) = current_gravity
                .filter(|_| self.gravity_projection_initialized && self.gravity_weight > 0.0)
            {
                let residual = Self::gravity_features(self.normalized_sample(sample), gravity)
                    .dot(&self.parameters)
                    - self.gravity_projection;
                // Gravity is optional: a failed update keeps the previous
                // statistic instead of zeroing the live quality.
                if let Some(mean_square) = Self::update_running_mean_square(
                    self.gravity_residual_mean_square,
                    residual * residual,
                    update_weight,
                ) {
                    self.gravity_residual_mean_square = Some(mean_square);
                }
            }
        }
        let coverage = self.mean_centered_coverage();
        let radial_fitness = Self::radial_fitness_score(self.radial_residual_mean_square);
        let gravity_fitness = Self::gravity_fitness_score(self.gravity_residual_mean_square);
        self.quality = CalibrationQuality::new(coverage, radial_fitness, gravity_fitness);
        Some(candidate)
    }

    fn sample(&self, row: usize) -> Vector3<f32> {
        self.sample_matrix.row(row).transpose().into_owned()
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
}

#[cfg(test)]
#[path = "mag_calibrator_test.rs"]
mod mag_calibrator_test;
