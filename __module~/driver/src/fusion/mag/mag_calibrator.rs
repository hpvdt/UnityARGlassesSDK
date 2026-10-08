use nalgebra::{DMatrix, DVector, Matrix3, SVector, Vector3};

use super::bad_mag_cause::{BadMagCause, BadReading};
use super::calibration_quality::CalibrationQuality;
use super::mag_model::{MagModel, CALIBRATION_PARAMETER_COUNT, SHAPE_REGULARIZATION};
use super::mag_samples::{ConcreteRow, Row};
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
const MIN_MAG_NORM: f32 = 0.4;
/// Default gravity-surrogate weight. With the preconditioned surrogate, the
/// projection target converges to the exact magnetic dip
/// `g_i^T m_i` (see `MagModel::gravity_frame`), so a retained gravity
/// direction should always inform the fit: the rotated-eigenvector
/// anisotropy sweep that regressed under the unpreconditioned surrogate now
/// improves in every case
/// (`mag_calibrator_gravity_surrogate_survives_strong_anisotropy`), and the
/// fixed-seed SimMotion benchmark at this weight matched the old disabled
/// baseline to within 0.04 degree. `0.01` keeps the gravity term a small
/// correction on the radial objective; [`MagCalibrator::gravity_weight`]
/// with `0` opts out.
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
/// Numerical floor of the optimizer feature-energy scales, also reused as
/// the floor of the squared gravity projection scale $\sigma_g^2$.
pub(super) const ONLINE_SCALE_EPSILON /*$\epsilon$*/: f32 = 1.0e-4;
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

/// Result of evaluating one FRD magnetometer observation.
/// All quality factors, including the learned dip projection `dip_sin`,
/// are exposed directly through [`Deref`] to [`CalibrationQuality`], so
/// `result.confidence()` reads the same as `result.quality.confidence()`.
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

/// Online regularized ellipsoid fit for a hard-iron offset and full SPD
/// soft-iron correction from a fixed, diverse sample buffer. Live quality
/// factors are recomputed from the retained rows on each quality update, so
/// only the online-optimizer parameters carry history beyond the cache.
pub struct MagCalibrator<const N: usize> {
    hard_iron_offset: Vector3<f32>,     /*$b$*/
    soft_iron_correction: Matrix3<f32>, /*$A$*/
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
    neighbor_count: usize, /*$k$*/
    max_sample_lifespan_us: u64,
    minibatch_size: usize, /*$|B|$*/
    replay_updates: usize,
    replay_minibatch_size: usize, /*$B_r$*/
    prng_state: u64,
    optimizer_steps: u64,
    publication_quality_streak: usize,
    /// The calibration model whose live quality is estimated and published:
    /// the retained sample cache, the online ellipsoid coefficients with the
    /// cache normalization they are expressed in, the gravity-projection
    /// state, and the live quality factors.
    model: MagModel<N>,
}

impl<const N: usize> Default for MagCalibrator<N> {
    fn default() -> Self {
        Self {
            hard_iron_offset: Vector3::zeros(),
            soft_iron_correction: Matrix3::identity(),
            calibration_initialized: false,
            mean_distance: Default::default(),
            neighbor_cache: [[NeighborEntry::EMPTY; NEIGHBOR_CACHE_CAPACITY]; N],
            neighbor_cache_len: [0; N],
            neighbor_count: 2, // Works well in testing
            max_sample_lifespan_us: 60 * 60 * 1_000_000,
            minibatch_size: DEFAULT_MINIBATCH_SIZE.min(N.max(1)),
            replay_updates: DEFAULT_REPLAY_UPDATES,
            replay_minibatch_size: DEFAULT_REPLAY_MINIBATCH_SIZE.min(N.max(1)),
            prng_state: ONLINE_PRNG_SEED,
            optimizer_steps: 0,
            publication_quality_streak: 0,
            model: MagModel {
                samples: Box::default(),
                parameters: MagModel::<N>::parameter_prior(),
                learned_gravity_projection: None,
                gravity_frame: Matrix3::identity(),
                gravity_weight: DEFAULT_GRAVITY_WEIGHT,
                quality: CalibrationQuality::ZERO,
            },
        }
    }
}

impl<const N: usize> MagCalibrator<N> {
    /// Create a new calibrator instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the number of nearest neighbors `k` whose mean distance
    /// scores each buffered row in the diversity heuristic.
    pub fn num_neighbors(self, neighbor_count: usize) -> Self {
        Self {
            neighbor_count: neighbor_count.clamp(1, N.saturating_sub(1).max(1)),
            ..self
        }
    }

    /// Configure the maximum time a sample remains in the calibration buffer,
    /// in microseconds. The default is one hour.
    ///
    /// Expiry strictly bounds cache membership and the live quality history:
    /// coverage and both loss statistics are recomputed from the retained
    /// rows on every quality update. Only the online-optimizer parameter
    /// history outlives the rows that produced it, diluting through the
    /// floored learning rate (see "Known adaptation limitation" in the fusion
    /// `AGENTS.md`).
    pub fn max_sample_lifespan_us(self, max_sample_lifespan_us: u64) -> Self {
        Self {
            max_sample_lifespan_us,
            ..self
        }
    }

    /// Configure the relative weight of the gravity-consistency residual.
    /// The default is `DEFAULT_GRAVITY_WEIGHT` (0.01): with the
    /// preconditioned surrogate the projection target converges to the exact
    /// magnetic dip, so a retained gravity direction always informs the fit.
    /// Pass `0` to opt out, reducing the objective and the live loss record
    /// to the purely radial terms even when rows carry gravity.
    pub fn gravity_weight(self, gravity_weight: f32) -> Self {
        Self {
            model: MagModel {
                gravity_weight: if gravity_weight.is_finite() {
                    gravity_weight.max(0.0)
                } else {
                    0.0
                },
                ..self.model
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
        // The normalization is derived once per minibatch: the raw moments
        // cannot change while the features are built, so this leaves every
        // observation's normalized sample bit-equal to a per-observation
        // `normalized_sample` call while dropping the fixed-size moment
        // recomputation that call performs.
        let (mean, rms_radius) = self.model.samples.stats.normalization();
        let observations = minibatch
            .current_sample
            .map(|sample| (sample, minibatch.current_gravity))
            .into_iter()
            .chain((0..minibatch.random_draws).map_while(|_| {
                Self::random_cache_row(
                    &mut random_state,
                    self.model.samples.stats.sample_row_count,
                    minibatch.accepted_row,
                )
                .map(|row| {
                    let row = self.model.samples.view(row);
                    (row.sample(), row.gravity())
                })
            }));
        let mut radial_rows = Vec::with_capacity(minibatch.random_draws + 1);
        let mut gravity_rows = Vec::with_capacity(minibatch.random_draws + 1);
        for (sample, gravity) in observations {
            let normalized = (sample - mean) / rms_radius;
            radial_rows.push(MagModel::<N>::features(normalized));
            if let Some(gravity) =
                gravity.filter(|_| self.model.learned_gravity_projection.is_some())
            {
                gravity_rows.push(MagModel::<N>::gravity_features(
                    normalized,
                    self.model.preconditioned_gravity(gravity),
                ));
            }
        }
        (
            Self::feature_matrix(&radial_rows),
            Self::feature_matrix(&gravity_rows),
            random_state,
        )
    }

    /// Minibatch objective with the gravity residual expressed relative to
    /// `gravity_scale`: $e_{g,i} = (\psi_i^T \theta - \kappa) / \sigma_g$.
    /// The scale is frozen for the duration of one optimizer update (it is
    /// computed from the pre-update parameters), so the evaluated objective
    /// stays a convex quadratic in $(\theta, \kappa)$. The feature matrices
    /// are built once per update by [`Self::apply_minibatch_update`], so the
    /// repeated evaluations of the bounded half-step search reuse the same
    /// sampled minibatch instead of resampling and rebuilding it.
    fn minibatch_objective(
        features: &DMatrix<f32>,
        gravity_features: &DMatrix<f32>,
        gravity_weight: f32,
        parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>,
        kappa: f32,
        gravity_scale: f32,
    ) -> f32 {
        let residuals = features * parameters - DVector::from_element(features.nrows(), 1.0);
        let mut objective = 0.5 * residuals.norm_squared() / features.nrows() as f32
            + MagModel::<N>::regularization_loss(parameters);
        if gravity_features.nrows() > 0 {
            let residuals = (gravity_features * parameters
                - DVector::from_element(gravity_features.nrows(), kappa))
                / gravity_scale;
            objective +=
                0.5 * gravity_weight * residuals.norm_squared() / gravity_features.nrows() as f32;
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
        if !self.model.samples.stats.sample_normalization_usable() {
            return;
        }
        self.model
            .initialize_gravity_projection(current_sample, current_gravity);

        let random_draws =
            if self.model.samples.stats.sample_row_count > usize::from(accepted_row.is_some()) {
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
        if self.calibration_initialized || self.model.samples.stats.sample_row_count == 0 {
            return;
        }
        let replay_count /*$p$*/ = self
            .replay_updates
            .saturating_mul(self.model.samples.stats.sample_row_count)
            / N.max(1);
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

        let parameters /*$\theta$*/ = self.model.parameters;
        // Gravity rows exist only once the projection is seeded.
        let kappa /*$\kappa$*/ = self.model.learned_gravity_projection.unwrap_or(0.0);
        let residuals = &features * parameters - DVector::from_element(features.nrows(), 1.0);
        let mut gradient /*$\nabla_\theta$*/ =
            Self::parameter_vector(features.tr_mul(&residuals)) / features.nrows() as f32;
        let mut gradient_scale /*$s_{\theta}$*/ =
            Self::parameter_vector(features.map(|value| value * value).row_sum_tr())
                / features.nrows() as f32;
        let mut kappa_gradient /*$\nabla_\kappa$*/ = 0.0;
        // Projection scale $\sigma_g$ of the gravity residual: the RMS
        // projection $\psi^T \theta$ over this minibatch, floored for
        // numerical safety and frozen for the whole update. Normalizing the
        // residual by $\sigma_g$ makes the data term measure the magnetic
        // dip's relative consistency instead of absolute equation units that
        // scale with the raw field radius: without it the gravity term's
        // effective pull and the reported loss scale would drift with the
        // device calibration state (the hint-noise residual of a real trace
        // scales with the sample radius $r$).
        let mut gravity_scale /*$\sigma_g$*/ = 1.0;
        if gravity_features.nrows() > 0 {
            let projections = &gravity_features * parameters;
            gravity_scale = (projections.norm_squared() / gravity_features.nrows() as f32)
                .max(ONLINE_SCALE_EPSILON)
                .sqrt();
            let gravity_scale_squared = gravity_scale * gravity_scale;
            let residuals = &gravity_features * parameters
                - DVector::from_element(gravity_features.nrows(), kappa);
            gradient += self.model.gravity_weight
                * Self::parameter_vector(gravity_features.tr_mul(&residuals))
                / (gravity_features.nrows() as f32 * gravity_scale_squared);
            gradient_scale += self.model.gravity_weight
                * Self::parameter_vector(gravity_features.map(|value| value * value).row_sum_tr())
                / (gravity_features.nrows() as f32 * gravity_scale_squared);
            kappa_gradient = -residuals.sum()
                * (self.model.gravity_weight
                    / (gravity_features.nrows() as f32 * gravity_scale_squared));
        }

        let prior = MagModel::<N>::parameter_prior();
        let regularization_weights =
            SVector::<f32, CALIBRATION_PARAMETER_COUNT>::from_row_slice(&[
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
            ]);
        gradient +=
            SHAPE_REGULARIZATION * regularization_weights.component_mul(&(parameters - prior));
        gradient_scale += SHAPE_REGULARIZATION * regularization_weights;
        gradient_scale.add_scalar_mut(ONLINE_SCALE_EPSILON);
        let descent_direction = gradient.component_div(&gradient_scale);
        let kappa_step = if gravity_features.nrows() > 0 && self.model.gravity_weight > 0.0 {
            // Curvature of the normalized gravity term in $\kappa$:
            // $w_g / \sigma_g^2$.
            kappa_gradient
                / (self.model.gravity_weight / (gravity_scale * gravity_scale)
                    + ONLINE_SCALE_EPSILON)
        } else {
            0.0
        };
        // The joint step covers $\theta$ and the scalar $\kappa$: `push`
        // widens the descent direction by the kappa step so the built-in
        // norm measures the whole bounded update.
        let descent_norm = descent_direction.push(kappa_step).norm();
        if !descent_norm.is_finite() || descent_norm <= f32::EPSILON {
            return false;
        }

        let old_objective = Self::minibatch_objective(
            &features,
            &gravity_features,
            self.model.gravity_weight,
            &parameters,
            kappa,
            gravity_scale,
        );
        let learning_rate = (ONLINE_INITIAL_LEARNING_RATE
            / (1.0 + self.optimizer_steps as f32 / ONLINE_LEARNING_RATE_DECAY_STEPS))
            .max(ONLINE_MIN_LEARNING_RATE);
        let mut step_size = learning_rate.min(ONLINE_MAX_STEP_NORM / descent_norm);
        for _ in 0..ONLINE_BACKTRACK_STEPS {
            let trial_parameters = parameters - step_size * descent_direction;
            let trial_kappa = kappa - step_size * kappa_step;
            let objective = Self::minibatch_objective(
                &features,
                &gravity_features,
                self.model.gravity_weight,
                &trial_parameters,
                trial_kappa,
                gravity_scale,
            );
            if trial_parameters.iter().all(|value| value.is_finite())
                && trial_kappa.is_finite()
                && objective.is_finite()
                && objective < old_objective
            {
                self.model.parameters = trial_parameters;
                // A gravity-free update leaves `trial_kappa` at the 0.0
                // placeholder; only an already seeded projection may be
                // refined, otherwise the placeholder would be planted as a
                // seed ahead of any gravity observation.
                if self.model.learned_gravity_projection.is_some() {
                    self.model.learned_gravity_projection = Some(trial_kappa);
                }
                return true;
            }
            step_size *= 0.5;
        }
        false
    }

    /// Computes squared distances from `mag_sample` to the first `count`
    /// rows of the sample buffer. Entries at and beyond `count` are set to
    /// infinity so selection never picks them. Each row's value is
    /// `(mag_sample - row_sample).norm_squared()`, assembled from the
    /// cache's bulk row-difference helper.
    fn squared_distances_to(&self, mag_sample: Vector3<f32>, count: usize) -> [f32; N] {
        let mut squared_distances = [f32::INFINITY; N];
        let mut differences = [Vector3::zeros(); N];
        self.model
            .samples
            .row_differences(mag_sample, count, &mut differences);
        for (j, dist) in squared_distances.iter_mut().enumerate().take(count) {
            *dist = differences[j].norm_squared();
        }
        squared_distances
    }

    /// Mean distance over the `k` smallest entries of `squared_distances`,
    /// selected in O(n) with a partial sort; `squared_distances` is reordered
    /// in the process. The square root is deferred until after selection, so
    /// only the `k` selected entries have their square roots taken. Returns
    /// infinity for `k == 0`.
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
        let squared_distances = self.squared_distances_to(self.model.samples.view(row).sample(), N);
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
        let mut squared_distances =
            self.squared_distances_to(self.model.samples.view(row).sample(), N);
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

    /// Returns the index of the buffered row with the lowest mean distance
    /// to its `k` nearest neighbors, derived from the incremental neighbor
    /// cache. Used to pick the victim row that a more diverse incoming
    /// sample may replace.
    fn lowest_mean_distance_by_index(&mut self) -> (usize, f32) {
        let neighbor_count = self.neighbor_count.min(N.saturating_sub(1));
        let mean_dist =
            SVector::<f32, N>::from_fn(|index, _| self.row_mean_distance(index, neighbor_count));
        self.mean_distance = mean_dist.mean();
        mean_dist.argmin()
    }

    /// Add a sample to the cache: the first `N` valid samples fill it
    /// unconditionally, and once the cache is full a sample is retained only
    /// if it scores more diverse than the least diverse retained row.
    ///
    /// `gravity_hint` is an optional co-timestamped body-frame FRD
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
        self.update_publication(valid_current_sample);
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
        // Old samples expire before the incoming magnetometer is validated,
        // so an invalid reading can still change retained support,
        // normalization, and live quality through expiry.
        let previous_count = self.model.samples.stats.sample_row_count;
        let mut index_map = [u32::MAX; N];
        self.model
            .samples
            .expire(timestamp_us, self.max_sample_lifespan_us, &mut index_map);
        if self.model.samples.stats.sample_row_count != previous_count {
            self.mean_distance = 0.0;
            self.remap_neighbor_cache(&index_map);
        }

        if !mag_sample.iter().all(|e| e.is_finite()) || mag_sample.norm_squared() <= f32::EPSILON {
            return false;
        }
        if N == 0 {
            return false;
        }
        let mut accepted_row = None;
        // Check if buffer is not yet "initialized" with real measurements
        if self.model.samples.stats.sample_row_count < N {
            let count = self.model.samples.stats.sample_row_count;
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
            self.model.samples.append(
                ConcreteRow::new(mag_sample, gravity_direction),
                timestamp_us,
            );
            self.reset_row_cache(count, &squared_distances, count);
            accepted_row = Some(count);
        }
        // Otherwise check which sample may be best to replace
        else {
            let neighbor_count = self.neighbor_count.min(N.saturating_sub(1));
            let (replacement_row, replacement_mean_distance) = self.lowest_mean_distance_by_index();
            let squared_distances = self.squared_distances_to(mag_sample, N);
            // Compare both scores against the rows retained after replacement.
            // Keep the original distances intact for neighbor-cache updates.
            let mut candidate_squared_distances = squared_distances;
            candidate_squared_distances[replacement_row] = f32::INFINITY;
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
                self.model.samples.replace(
                    replacement_row,
                    ConcreteRow::new(mag_sample, gravity_direction),
                    timestamp_us,
                );
                self.reset_row_cache(replacement_row, &squared_distances, N);
                accepted_row = Some(replacement_row);
            }
        }
        self.update_online_optimizer(mag_sample, gravity_direction, accepted_row);
        true
    }

    /// Get the mean of the per-row nearest-neighbor distances over the
    /// retained samples.
    pub fn get_mean_distance(&self) -> f32 {
        self.mean_distance
    }

    /// Returns the current bounded calibration quality in `[0, 1]`.
    ///
    /// Zero means the current working candidate is pending or unusable. A
    /// previously published correction can remain available while this value
    /// is zero after a rejected later candidate.
    pub fn get_confidence(&self) -> f32 {
        self.model.quality.confidence()
    }

    /// Calibrates a magnetometer vector that has already been converted to FRD.
    ///
    /// `gravity_hint` is an optional co-timestamped body-frame FRD
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
            return Ok(MagCalibrationResult {
                quality: self.model.quality,
                direction: None,
            });
        }
        let mut mag = self.soft_iron_correction * (raw_mag - self.hard_iron_offset);

        let mag_norm = mag.normalize_mut();
        if !mag_norm.is_finite() || mag_norm < MIN_MAG_NORM {
            Err(BadMagCause::BadReading(BadReading::WeakCalibratedReading {
                norm: mag_norm,
                min_norm: MIN_MAG_NORM,
            }))
        } else {
            Ok(MagCalibrationResult {
                quality: self.model.quality,
                direction: Some(mag),
            })
        }
    }

    fn update_publication(&mut self, current_sample_valid: bool) {
        let candidate = self.model.update_quality();
        if !current_sample_valid || candidate.is_none() {
            // Invalid observations and unusable candidates always reset the
            // streak: they are evidence against publishing, not jitter.
            self.publication_quality_streak = 0;
        } else if self.model.quality.confidence() >= MIN_PUBLICATION_CONFIDENCE {
            self.publication_quality_streak = self.publication_quality_streak.saturating_add(1);
        } else if self.model.quality.confidence() < PUBLICATION_STREAK_RESET_CONFIDENCE {
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
}

#[cfg(test)]
#[path = "mag_calibrator_test.rs"]
mod mag_calibrator_test;
