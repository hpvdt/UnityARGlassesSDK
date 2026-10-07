use nalgebra::{Matrix3, SVector, UnitQuaternion, Vector3};

use super::super::mag_model::{
    CoverageGramMatrix, MagModel, CALIBRATION_PARAMETER_COUNT, SHAPE_PRIOR_SCALE,
    SHAPE_REGULARIZATION,
};
use super::super::mag_samples::Row;
use super::super::BadMagCause;
use super::{
    MagCalibrationResult, MagCalibrator, MinibatchSpec, MIN_PUBLICATION_CONFIDENCE,
    MIN_PUBLICATION_STREAK, ONLINE_SCALE_EPSILON,
};

impl<const N: usize> MagCalibrator<N> {
    /// Independently recomputes the two terms of the radial online
    /// objective of the current working parameters: the mean square of the
    /// algebraic residual `phi(u_i)^T theta - 1` over the retained cache in
    /// normalized cache coordinates, row by row in ascending order, and the
    /// shape-regularization loss `0.5 * lambda * ||Q - c * I||_F^2`.
    /// Deliberately duplicates the accumulation inside `update_quality`
    /// instead of calling it, so the tests cross-check the production path.
    /// Gated on a valid working candidate to mirror the gating in
    /// `update_quality`.
    fn radial_objective_for_test(&self) -> Option<(f32, f32)> {
        self.model.working_candidate().ok()?;
        let mut sum = 0.0f32;
        for row in 0..self.model.samples.stats.sample_row_count {
            let residual = MagModel::<N>::features(
                self.model
                    .samples
                    .stats
                    .normalized_sample(self.model.samples.view(row).sample()),
            )
            .dot(&self.model.parameters)
                - 1.0;
            sum += residual * residual;
        }
        let (shape, _) = MagModel::<N>::unpack_ellipsoid_coefficients(&self.model.parameters);
        let regularization_loss = 0.5
            * SHAPE_REGULARIZATION
            * (shape - Matrix3::identity() * SHAPE_PRIOR_SCALE).norm_squared();
        Some((
            sum / self.model.samples.stats.sample_row_count as f32,
            regularization_loss,
        ))
    }

    /// Independently recomputes the gravity mean square of the current
    /// working parameters over the retained rows carrying a gravity
    /// direction, mirroring the gating and the projection-scale
    /// normalization of `update_quality`. The preconditioner frame is NOT
    /// refreshed here: the mirror intentionally reuses the frame state of
    /// the last production `update_quality`, so the comparison only makes
    /// sense after a production quality update (which `evaluate_correct`
    /// always performs).
    fn gravity_mean_square_for_test(&self) -> Option<f32> {
        let kappa = self.model.learned_gravity_projection?;
        if self.model.gravity_weight <= 0.0 {
            return None;
        }
        let mut sum = 0.0f32;
        let mut projection_square_sum = 0.0f32;
        let mut count = 0usize;
        for row in 0..self.model.samples.stats.sample_row_count {
            let row = self.model.samples.view(row);
            if let Some(gravity) = row.gravity() {
                let projection = MagModel::<N>::gravity_features(
                    self.model.samples.stats.normalized_sample(row.sample()),
                    self.model.preconditioned_gravity(gravity),
                )
                .dot(&self.model.parameters);
                let residual = projection - kappa;
                sum += residual * residual;
                projection_square_sum += projection * projection;
                count += 1;
            }
        }
        (count > 0).then(|| {
            let scale_squared = (projection_square_sum / count as f32).max(ONLINE_SCALE_EPSILON);
            sum / count as f32 / scale_squared
        })
    }

    fn working_quality_components(&self) -> (bool, f32, f32, f32) {
        if self.model.working_candidate().is_err() {
            return (false, 0.0, 0.0, 0.0);
        }
        let coverage = self.mean_centered_coverage_for_test();
        let radial_loss = self
            .radial_objective_for_test()
            .map_or(0.0, |(mean_square, regularization_loss)| {
                0.5 * mean_square + regularization_loss
            });
        let gravity_loss = self
            .gravity_mean_square_for_test()
            .map_or(0.0, |mean_square| {
                0.5 * self.model.gravity_weight * mean_square
            });
        (true, coverage, radial_loss, gravity_loss)
    }

    /// Independently recomputes the E-optimality coverage of the retained
    /// rows, mirroring the mean-centered Gram accumulation of the production
    /// quality pass row by row. Deliberately duplicates the accumulation
    /// instead of calling production state, so the tests cross-check the
    /// production path.
    fn mean_centered_coverage_for_test(&self) -> f32 {
        let (mu, _) = self.model.samples.stats.normalization();
        let mut directions = Vec::new();
        for row in 0..self.model.samples.stats.sample_row_count {
            let centered = self.model.samples.view(row).sample() - mu;
            if let Some(direction) = centered.try_normalize(f32::EPSILON) {
                directions.push(direction);
            }
        }
        MagModel::<N>::coverage_from_gram(
            &Self::coverage_gram_sum_for_test(&directions),
            self.model.samples.stats.sample_row_count,
        )
    }

    fn correct_working_for_test(&self, raw_mag: Vector3<f32>) -> Option<Vector3<f32>> {
        let candidate = self.model.working_candidate().ok()?;
        let corrected = candidate.correction * (raw_mag - candidate.offset);
        let norm = corrected.norm();
        (norm.is_finite() && norm > f32::EPSILON).then(|| corrected / norm)
    }

    fn publication_quality_streak_for_test(&self) -> usize {
        self.publication_quality_streak
    }

    fn coverage_scores_for_test(gram_sum: &CoverageGramMatrix, sample_row_count: usize) -> f32 {
        MagModel::<N>::coverage_from_gram(gram_sum, sample_row_count)
    }

    fn coverage_gram_sum_for_test(directions: &[Vector3<f32>]) -> CoverageGramMatrix {
        let mut gram_sum = CoverageGramMatrix::zeros();
        for &direction in directions {
            let feature = MagModel::<N>::coverage_feature(direction);
            gram_sum += feature * feature.transpose();
        }
        gram_sum
    }

    fn raw_moments_for_test(&self) -> (usize, Vector3<f64>, Matrix3<f64>) {
        (
            self.model.samples.stats.sample_row_count,
            self.model.samples.stats.raw_sample_sum,
            self.model.samples.stats.raw_outer_product_sum,
        )
    }

    /// Verifies maintained raw moments against a direct current-cache sum.
    fn check_raw_moments(&self) -> Result<(), String> {
        let (sum, outer_sum) = (0..self.model.samples.stats.sample_row_count).fold(
            (Vector3::<f64>::zeros(), Matrix3::<f64>::zeros()),
            |(sum, outer_sum), row| {
                let sample = self.model.samples.view(row).sample().cast::<f64>();
                (sum + sample, outer_sum + sample * sample.transpose())
            },
        );
        let scale = sum.norm().max(outer_sum.norm()).max(1.0);
        let error = (self.model.samples.stats.raw_sample_sum - sum)
            .norm()
            .max((self.model.samples.stats.raw_outer_product_sum - outer_sum).norm());
        if error <= 1.0e-12 * scale {
            Ok(())
        } else {
            Err(format!("raw moment error={error} scale={scale}"))
        }
    }

    /// Verifies the neighbor-cache invariant against the current buffer
    /// contents: for every buffered row, the cached entries must reference
    /// distinct live rows with exactly matching squared distances, be sorted
    /// ascending, and their distance values must equal the `len` smallest
    /// true distances to the row's other buffered rows. Read-only; used by
    /// tests to cross-check the incremental cache maintenance.
    fn check_neighbor_cache(&self) -> Result<(), String> {
        for row in 0..self.model.samples.stats.sample_row_count {
            let len = self.neighbor_cache_len[row] as usize;
            let cache = &self.neighbor_cache[row][..len];
            let mut true_dists: Vec<f32> = (0..self.model.samples.stats.sample_row_count)
                .filter(|&j| j != row)
                .map(|j| {
                    let diff =
                        self.model.samples.view(row).sample() - self.model.samples.view(j).sample();
                    diff.dot(&diff)
                })
                .collect();
            true_dists.sort_unstable_by(|a, b| a.total_cmp(b));
            for (i, entry) in cache.iter().enumerate() {
                if entry.row as usize >= self.model.samples.stats.sample_row_count
                    || entry.row as usize == row
                {
                    return Err(format!("row {row}: entry {i} references row {}", entry.row));
                }
                if i > 0 && cache[i - 1].squared_distance > entry.squared_distance {
                    return Err(format!("row {row}: entry {i} out of order"));
                }
                if cache[..i].iter().any(|e| e.row == entry.row) {
                    return Err(format!("row {row}: duplicate entry for row {}", entry.row));
                }
                let diff = self.model.samples.view(row).sample()
                    - self.model.samples.view(entry.row as usize).sample();
                if diff.dot(&diff) != entry.squared_distance {
                    return Err(format!("row {row}: stale distance for row {}", entry.row));
                }
                if true_dists.get(i) != cache.get(i).map(|e| &e.squared_distance) {
                    return Err(format!(
                        "row {row}: entry {i} is not the true {}-nearest neighbor",
                        i + 1
                    ));
                }
            }
            if len > true_dists.len() {
                return Err(format!("row {row}: cache longer than the neighbor pool"));
            }
        }
        Ok(())
    }
}

#[test]
fn mag_calibrator_corrects_synthetic_full_spd_distortion() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<63>(offset, distortion);

    for (timestamp_us, expected) in [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(1.0, -2.0, 3.0).normalize(),
    ]
    .into_iter()
    .enumerate()
    {
        let corrected = calibrated(calibrator.evaluate_correct(
            offset + distortion * expected,
            None,
            timestamp_us as u64,
        ));
        assert_vec_close(corrected, expected, 0.05);
    }
}

#[test]
fn mag_calibrator_corrects_asymmetrically_sampled_distortion() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = MagCalibrator::<63>::new();
    // Non-uniform spiral covering the sphere: the E-optimality coverage score
    // refuses publication for a sweep that never visits a cap of the sphere,
    // so the asymmetric sampling must still span all directions.
    for i in 0..63 {
        let azimuth = 0.37 + i as f32 * 1.21;
        let z = -0.85 + 1.7 * i as f32 / 62.0;
        let xy_radius = (1.0 - z * z).sqrt();
        let direction = Vector3::new(xy_radius * azimuth.cos(), xy_radius * azimuth.sin(), z);
        let _ = calibrator.evaluate_correct(offset + distortion * direction, None, i as u64);
    }

    for i in 0..16 * 63 {
        let sample_index = i % 63;
        let azimuth = 0.37 + sample_index as f32 * 1.21;
        let z = -0.85 + 1.7 * sample_index as f32 / 62.0;
        let xy_radius = (1.0 - z * z).sqrt();
        let direction = Vector3::new(xy_radius * azimuth.cos(), xy_radius * azimuth.sin(), z);
        let _ = calibrator.evaluate_correct(offset + distortion * direction, None, (64 + i) as u64);
    }
    let expected = Vector3::new(1.0, -2.0, -1.0).normalize();
    let corrected =
        calibrated(calibrator.evaluate_correct(offset + distortion * expected, None, 64 + 16 * 63));

    assert_vec_close(corrected, expected, 0.05);
}

#[test]
fn mag_calibrator_returns_stable_online_corrections() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<63>(offset, distortion);
    let expected = Vector3::new(1.0, -2.0, 3.0).normalize();
    let raw = offset + distortion * expected;

    let first = calibrated(calibrator.evaluate_correct(raw, None, 1));
    let second = calibrated(calibrator.evaluate_correct(raw, None, 2));

    assert_vec_close(second, first, 0.01);
}

#[test]
fn mag_calibrator_stays_pending_with_underconstrained_or_degenerate_data() {
    let mut calibrator = MagCalibrator::<9>::new();
    let sample = Vector3::new(5.0, 6.0, 7.0);
    let single = calibrator.evaluate_correct(sample, None, 0);

    assert!(matches!(
        single,
        Ok(MagCalibrationResult {
            quality,
            direction: None,
            ..
        }) if quality.confidence() == 0.0
    ));
    let result = (1..9)
        .map(|timestamp_us| calibrator.evaluate_correct(sample, None, timestamp_us))
        .last()
        .unwrap();

    assert!(matches!(
        result,
        Ok(MagCalibrationResult {
            quality,
            direction: None,
            ..
        }) if quality.confidence() == 0.0
    ));
    assert_eq!(calibrator.get_confidence(), 0.0);
}

#[test]
fn mag_calibrator_publishes_before_the_buffer_is_full() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = MagCalibrator::<1023>::new();
    let mut published_at = None;
    let mut qualifying_streak = 0;
    for i in 0..1022 {
        let direction = sample_direction(i % 63, 63);
        let result = calibrator
            .evaluate_correct(offset + distortion * direction, None, i as u64)
            .unwrap();
        if result.confidence() >= MIN_PUBLICATION_CONFIDENCE {
            qualifying_streak += 1;
        } else {
            qualifying_streak = 0;
        }
        if result.direction.is_some() {
            assert!(qualifying_streak >= MIN_PUBLICATION_STREAK);
            published_at = Some(i + 1);
            break;
        }
    }

    let components = calibrator.working_quality_components();
    let published_at = published_at.unwrap_or_else(|| {
        panic!("partial cache never published a valid correction: components={components:?}")
    });
    assert!(published_at >= 9);
    assert!(published_at < 1023);
    assert!(calibrator.get_confidence() >= MIN_PUBLICATION_CONFIDENCE);
}

#[test]
fn mag_calibrator_resets_publication_streak_after_invalid_sample() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = MagCalibrator::<1023>::new();

    for i in 0..1022 {
        let direction = sample_direction(i % 63, 63);
        let _ = calibrator.evaluate_correct(offset + distortion * direction, None, i as u64);
        if calibrator.publication_quality_streak_for_test() > 0 {
            let _ = calibrator.evaluate_correct(Vector3::repeat(f32::NAN), None, i as u64 + 1);
            assert_eq!(calibrator.publication_quality_streak_for_test(), 0);
            return;
        }
    }

    panic!("confidence never began a publication-quality streak");
}

#[test]
fn mag_calibrator_rejects_nearly_collinear_samples() {
    let mut calibrator = MagCalibrator::<12>::new();
    let mut result = None;
    for i in 0..12 {
        let t = i as f32 * 0.0001;
        result = Some(calibrator.evaluate_correct(
            Vector3::new(10.0 + t, -5.0 + 2.0 * t, 3.0 + 0.5 * t),
            None,
            i as u64,
        ));
    }

    assert!(matches!(
        result.unwrap(),
        Ok(MagCalibrationResult {
            quality,
            direction: None,
            ..
        }) if quality.confidence() == 0.0
    ));
}

#[test]
fn mag_calibrator_keeps_last_correction_after_rejected_refit() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<63>(offset, distortion).max_sample_lifespan_us(0);
    let mut result = None;
    let expected = Vector3::x();
    let raw = offset + distortion * expected;

    for _ in 0..63 {
        result = Some(calibrator.evaluate_correct(raw, None, 1));
    }

    let result = result.unwrap().unwrap();
    assert_eq!(result.confidence(), 0.0);
    assert_vec_close(
        result
            .direction
            .expect("last published correction was discarded"),
        expected,
        0.05,
    );
    // Working state is never rebased or reset, and the radial statistic is
    // recomputed from the retained rows: the expired history refilled with
    // identical samples leaves a zero-radius normalization, so no usable
    // candidate exists to score.
    assert_eq!(calibrator.radial_objective_for_test(), None);
}

#[test]
fn mag_calibrator_loss_recovers_after_full_expiry() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<63>(offset, distortion).max_sample_lifespan_us(0);
    let expected = Vector3::x();
    let raw = offset + distortion * expected;

    // A strictly larger timestamp with a zero lifespan expires every
    // retained row, emptying the cache before the new sample lands. The
    // last published correction stays in use even though live confidence
    // collapses.
    let result = calibrator.evaluate_correct(raw, None, 1).unwrap();
    assert_eq!(result.confidence(), 0.0);
    assert_vec_close(
        result
            .direction
            .expect("last published correction was discarded"),
        expected,
        0.05,
    );

    // The statistic that produced the pre-expiry loss is gone with the
    // expired rows: while fewer than nine fresh rows are retained, live
    // quality stays explicitly pending at zero.
    for i in 0..7 {
        let raw = offset + distortion * sample_direction(i, 63);
        let result = calibrator.evaluate_correct(raw, None, 1).unwrap();
        assert_eq!(
            result.confidence(),
            0.0,
            "fresh row {} escaped pending",
            i + 2
        );
    }

    // Once enough fresh rows are retained, the reported losses are exactly
    // the online objectives recomputed over the calibrator's own retained
    // cache with its current working parameters — the radial loss as half
    // the mean square algebraic residual plus the shape-regularization
    // loss; nothing from the expired rows survives in them.
    let mut result = None;
    for i in 7..63 {
        let raw = offset + distortion * sample_direction(i, 63);
        result = Some(calibrator.evaluate_correct(raw, None, 1).unwrap());
    }
    let result = result.unwrap();
    let (radial_mean_square, regularization_loss) = calibrator
        .radial_objective_for_test()
        .expect("recovered cache produced no working candidate");
    assert_eq!(
        result.radial_loss,
        0.5 * radial_mean_square + regularization_loss
    );
    assert_eq!(result.regularization_loss, regularization_loss);
    // No gravity direction was ever supplied, so the objective contains no
    // gravity term.
    assert_eq!(result.gravity_loss, 0.0);
}

#[test]
fn mag_calibrator_loss_depends_only_on_retained_rows() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let sample = |i: usize| offset + distortion * sample_direction(i, 63);
    let gravity = |i: usize| sample_direction(i + 31, 63);

    // Calibrator A ingests an early batch whose older rows then expire
    // (lifespan 100 us, final batch timestamp 153 keeps exactly the rows
    // with timestamp >= 53), followed by a fresh batch. Both calibrators
    // use the default weight, so the gravity-projection statistic stays
    // live.
    let mut lifespan_a = MagCalibrator::<63>::new().max_sample_lifespan_us(100);
    for timestamp_us in 50..=59 {
        let index = timestamp_us as usize - 50;
        lifespan_a.evaluate_sample_vec(sample(index), Some(gravity(index)), timestamp_us);
    }
    for timestamp_us in 142..=153 {
        let index = timestamp_us as usize - 100;
        lifespan_a.evaluate_sample_vec(sample(index), Some(gravity(index)), timestamp_us);
    }

    // Calibrator B ingests only the rows that survive in A, in the same
    // order. Its optimizer history differs from A's (B never saw the
    // expired prefix), which is the non-strict-by-design part; only the
    // caches and the cache-derived loss semantics are pinned here.
    let mut survivors_only = MagCalibrator::<63>::new().max_sample_lifespan_us(100);
    for timestamp_us in 53..=59 {
        let index = timestamp_us as usize - 50;
        survivors_only.evaluate_sample_vec(sample(index), Some(gravity(index)), timestamp_us);
    }
    for timestamp_us in 142..=153 {
        let index = timestamp_us as usize - 100;
        survivors_only.evaluate_sample_vec(sample(index), Some(gravity(index)), timestamp_us);
    }

    // The expired prefix is gone from A's cache: both caches hold exactly
    // the surviving rows in stable insertion order.
    assert_eq!(lifespan_a.model.samples.stats.sample_row_count, 19);
    assert_caches_identical(&lifespan_a, &survivors_only);

    // A shared suffix at a fixed timestamp (no further expiry) lets both
    // working candidates converge; cache ingestion is parameter-independent,
    // so identical inputs keep the caches identical.
    let mut result_a = None;
    for i in 0..2 * 63 {
        let result = lifespan_a.evaluate_correct(sample(i % 63), Some(gravity(i % 63)), 153);
        survivors_only.evaluate_sample_vec(sample(i % 63), Some(gravity(i % 63)), 153);
        result_a = Some(result);
    }
    assert_caches_identical(&lifespan_a, &survivors_only);

    // A's reported losses are pure functions of its retained rows and its
    // current working parameters, recomputed independently here row by row.
    // The losses themselves are NOT asserted bitwise equal to B's: A and B
    // share the cache but not the online-optimizer parameter history, which
    // legitimately still carries the expired rows' gradients (non-strict by
    // design; see "Known adaptation limitation" in the fusion AGENTS.md).
    let result_a = result_a.unwrap().unwrap();
    let (radial_mean_square, regularization_loss) = lifespan_a
        .radial_objective_for_test()
        .expect("retained cache produced no working candidate");
    assert_eq!(
        result_a.radial_loss,
        0.5 * radial_mean_square + regularization_loss
    );
    assert_eq!(result_a.regularization_loss, regularization_loss);
    let gravity_mean_square = lifespan_a
        .gravity_mean_square_for_test()
        .expect("retained cache carried no gravity statistic");
    assert_eq!(
        result_a.gravity_loss,
        0.5 * lifespan_a.model.gravity_weight * gravity_mean_square
    );
}

#[test]
fn mag_calibrator_radial_loss_reports_the_full_objective() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);

    // Pending quality (fewer than nine retained rows) reports no loss
    // statistics.
    let mut calibrator = MagCalibrator::<63>::new();
    let result = calibrator
        .evaluate_correct(offset + distortion * sample_direction(0, 63), None, 0)
        .unwrap();
    assert_eq!(result.confidence(), 0.0);
    assert_eq!(result.regularization_loss, 0.0);
    assert_eq!(result.radial_loss, 0.0);

    // Once the working candidate has moved off the prior, the
    // regularization loss is positive and the radial loss reports the
    // full radial objective — half the data residual mean square plus the
    // loss — so it is strictly above what a data-only record would report,
    // just as the objective the optimizer descends exceeds its data term.
    let mut calibrator = seeded_calibrator::<63>(offset, distortion);
    let result = calibrator
        .evaluate_correct(offset + distortion * sample_direction(0, 63), None, 1)
        .unwrap();
    let (radial_mean_square, regularization_loss) = calibrator
        .radial_objective_for_test()
        .expect("seeded calibrator produced no working candidate");
    assert!(regularization_loss > 0.0);
    assert_eq!(result.regularization_loss, regularization_loss);
    let data_only = 0.5 * radial_mean_square;
    let combined = 0.5 * radial_mean_square + regularization_loss;
    assert!(combined > data_only);
    assert_eq!(result.radial_loss, combined);
}

/// Asserts that two calibrators retain exactly the same rows with the same
/// timestamps and gravity directions, in the same order.
fn assert_caches_identical<const N: usize>(first: &MagCalibrator<N>, second: &MagCalibrator<N>) {
    assert_eq!(
        first.model.samples.stats.sample_row_count,
        second.model.samples.stats.sample_row_count
    );
    for row in 0..first.model.samples.stats.sample_row_count {
        assert_eq!(
            first.model.samples.view(row).sample(),
            second.model.samples.view(row).sample()
        );
        assert_eq!(
            first.model.samples.timestamps_us[row],
            second.model.samples.timestamps_us[row]
        );
        assert_eq!(
            first.model.samples.view(row).gravity(),
            second.model.samples.view(row).gravity()
        );
    }
}

#[test]
fn mag_calibrator_clamps_neighbor_count_through_public_result() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let expected = Vector3::new(1.0, -2.0, 3.0).normalize();

    for k in [0, 99] {
        let mut calibrator = seeded_calibrator::<63>(offset, distortion).num_neighbors(k);
        let corrected =
            calibrated(calibrator.evaluate_correct(offset + distortion * expected, None, 1));
        assert_vec_close(corrected, expected, 0.05);
    }
}

#[test]
fn mag_calibrator_clamps_minibatch_size_and_is_deterministic() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let expected = Vector3::new(1.0, -2.0, 3.0).normalize();
    let raw = offset + distortion * expected;

    for minibatch_size in [0, usize::MAX] {
        let mut first = MagCalibrator::<63>::new().minibatch_size(minibatch_size);
        let mut second = MagCalibrator::<63>::new().minibatch_size(minibatch_size);
        for i in 0..17 * 63 {
            let direction = sample_direction(i % 63, 63);
            let raw = offset + distortion * direction;
            assert_eq!(
                first.evaluate_correct(raw, None, i as u64),
                second.evaluate_correct(raw, None, i as u64)
            );
        }
        let first_result = first.evaluate_correct(raw, None, 10_000);
        let second_result = second.evaluate_correct(raw, None, 10_000);
        assert_eq!(first_result, second_result);
        if minibatch_size == usize::MAX {
            assert_vec_close(calibrated(first_result), expected, 0.05);
        }
    }
}

#[test]
fn mag_calibrator_converges_faster_with_cache_replay() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut replayed = MagCalibrator::<63>::new();
    let mut plain = MagCalibrator::<63>::new().replay_updates(0);

    let probe_error = |calibrator: &MagCalibrator<63>| {
        [
            Vector3::x(),
            Vector3::y(),
            Vector3::z(),
            Vector3::new(1.0, -2.0, 3.0).normalize(),
        ]
        .into_iter()
        .try_fold(0.0, |sum, expected| {
            calibrator
                .correct_working_for_test(offset + distortion * expected)
                .map(|actual| sum + (actual - expected).norm())
        })
    };

    let mut replayed_published_at = None;
    let mut plain_published_at = None;
    let mut replayed_converged_at = None;
    let mut plain_converged_at = None;
    for i in 0..16 * 63 {
        let raw = offset + distortion * sample_direction(i % 63, 63);
        let replayed_result = replayed.evaluate_correct(raw, None, i as u64).unwrap();
        let plain_result = plain.evaluate_correct(raw, None, i as u64).unwrap();
        if replayed_result.direction.is_some() && replayed_published_at.is_none() {
            replayed_published_at = Some(i);
        }
        if plain_result.direction.is_some() && plain_published_at.is_none() {
            plain_published_at = Some(i);
        }
        if replayed_converged_at.is_none()
            && probe_error(&replayed).is_some_and(|error| error < 0.02)
        {
            replayed_converged_at = Some(i);
        }
        if plain_converged_at.is_none() && probe_error(&plain).is_some_and(|error| error < 0.02) {
            plain_converged_at = Some(i);
        }
    }
    let replayed_published_at = replayed_published_at.expect("replayed calibration stayed pending");
    let plain_published_at = plain_published_at.expect("plain calibration stayed pending");
    let replayed_converged_at =
        replayed_converged_at.expect("replayed calibration did not converge");
    let plain_converged_at = plain_converged_at.expect("plain calibration did not converge");
    assert!(
        replayed_converged_at < plain_converged_at,
        "replayed_converged_at={replayed_converged_at} plain_converged_at={plain_converged_at}"
    );
    assert!(replayed_published_at <= plain_published_at);

    let replayed_error = probe_error(&replayed).unwrap();
    let plain_error = probe_error(&plain).unwrap();

    assert!(replayed_error < 0.02, "replayed_error={replayed_error}");
    assert!(plain_error < 0.02, "plain_error={plain_error}");
}

#[test]
fn mag_calibrator_clamps_replay_configuration_and_is_deterministic() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let expected = Vector3::new(1.0, -2.0, 3.0).normalize();
    let raw = offset + distortion * expected;

    for (replay_updates, replay_minibatch_size) in [(0, 0), (3, 0), (3, usize::MAX)] {
        let configure = || {
            MagCalibrator::<63>::new()
                .replay_updates(replay_updates)
                .replay_minibatch_size(replay_minibatch_size)
        };
        let mut first = configure();
        let mut second = configure();
        for i in 0..17 * 63 {
            let direction = sample_direction(i % 63, 63);
            let raw = offset + distortion * direction;
            assert_eq!(
                first.evaluate_correct(raw, None, i as u64),
                second.evaluate_correct(raw, None, i as u64)
            );
        }
        let first_result = first.evaluate_correct(raw, None, 10_000);
        let second_result = second.evaluate_correct(raw, None, 10_000);
        assert_eq!(first_result, second_result);
        // A single-observation replay minibatch is degenerate, like a
        // single-observation anchored minibatch, so only configurations with
        // enough observations per update are required to converge.
        if replay_updates == 0 || replay_minibatch_size == usize::MAX {
            assert_vec_close(calibrated(first_result), expected, 0.05);
        }
    }
}

#[test]
fn mag_calibrator_accepts_zero_components_and_rejects_bad_vectors() {
    let mut calibrator = MagCalibrator::<12>::new();
    for (timestamp_us, sample) in [
        Vector3::new(45.0, 0.0, -12.0),
        Vector3::new(f32::NAN, 1.0, 1.0),
        Vector3::new(f32::INFINITY, 1.0, 1.0),
        Vector3::new(1.0e-8, 0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let result = calibrator.evaluate_correct(sample, None, timestamp_us as u64);
        assert!(matches!(
            result,
            Ok(MagCalibrationResult {
                quality,
                direction: None,
                ..
            }) if quality.confidence() == 0.0
        ));
    }
}

#[test]
fn mag_calibrator_defaults_sample_lifespan_to_one_hour() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<12>(offset, distortion);

    let result = calibrator.evaluate_correct(
        Vector3::new(20.0, 30.0, 40.0),
        None,
        60 * 60 * 1_000_000 + 1,
    );

    let result = result.unwrap();
    assert!(matches!(
        result,
        MagCalibrationResult {
            quality,
            direction: Some(_),
            ..
        } if quality.confidence() == 0.0
    ));
}

#[test]
fn mag_calibrator_uses_configured_sample_lifespan() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<12>(offset, distortion).max_sample_lifespan_us(10);

    let result = calibrator.evaluate_correct(Vector3::new(20.0, 30.0, 40.0), None, 11);

    let result = result.unwrap();
    assert!(matches!(
        result,
        MagCalibrationResult {
            quality,
            direction: Some(_),
            ..
        } if quality.confidence() == 0.0
    ));
}

#[test]
fn mag_calibrator_improves_with_consistent_gravity() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
    let mut plain = MagCalibrator::<63>::new();
    // The default weight keeps the gravity surrogate active.
    let mut gravity_refined = MagCalibrator::<63>::new();

    for i in 0..63 {
        let attitude = UnitQuaternion::from_euler_angles(
            0.25 * (i as f32 * 0.7).sin(),
            0.35 * (i as f32 * 1.7).sin(),
            i as f32 * 2.4,
        );
        let body_mag = attitude.inverse() * world_mag;
        let body_gravity = attitude.inverse() * world_gravity;
        let raw = offset + distortion * body_mag;
        let _ = plain.evaluate_correct(raw, None, i as u64);
        let _ = gravity_refined.evaluate_correct(raw, Some(body_gravity), i as u64);
    }

    let probes = [
        Vector3::x(),
        Vector3::y(),
        Vector3::z(),
        Vector3::new(1.0, -2.0, 3.0).normalize(),
    ];
    let plain_error: f32 = probes
        .iter()
        .map(|&expected| {
            (plain
                .correct_working_for_test(offset + distortion * expected)
                .expect("plain working calibration is invalid")
                - expected)
                .norm()
        })
        .sum();
    let refined_error: f32 = probes
        .iter()
        .map(|&expected| {
            (gravity_refined
                .correct_working_for_test(offset + distortion * expected)
                .expect("gravity-refined working calibration is invalid")
                - expected)
                .norm()
        })
        .sum();

    assert!(
        refined_error < plain_error,
        "plain_error={plain_error} refined_error={refined_error}"
    );
}

#[test]
fn mag_calibrator_gravity_surrogate_survives_strong_anisotropy() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let diagonal = |x: f32, y: f32, z: f32| Matrix3::new(x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z);
    let rotate = |roll: f32, pitch: f32, yaw: f32, distortion: Matrix3<f32>| {
        let basis = UnitQuaternion::from_euler_angles(roll, pitch, yaw)
            .to_rotation_matrix()
            .into_inner();
        basis * distortion * basis.transpose()
    };
    // Strongly anisotropic SPD soft iron, with and without rotated
    // eigenvectors, spanning a range of condition numbers.
    let distortions = [
        diagonal(1.6, 0.6, 1.2),
        rotate(0.4, -0.5, 0.8, diagonal(1.7, 0.55, 1.25)),
        rotate(0.7, 0.2, -0.5, diagonal(1.5, 0.7, 1.6)),
    ];
    // Several physical magnetic dip angles relative to world gravity. Both
    // world directions are co-rotated into the body frame, so the dip angle
    // is constant and the gravity hint is physically consistent.
    let dip_cases = [
        (Vector3::new(0.8, 0.1, 0.5).normalize(), Vector3::z()),
        (
            Vector3::new(0.45, 0.25, 0.85).normalize(),
            Vector3::new(0.2, 0.1, 0.97).normalize(),
        ),
        (
            Vector3::new(0.25, 0.68, 0.42).normalize(),
            Vector3::new(0.1, -0.35, 0.9).normalize(),
        ),
    ];
    // Chord distance for unit vectors is ~angle in radians for small errors.
    // Aggregate the summed probe error across every anisotropic and dip case
    // before comparing: with the unpreconditioned surrogate the aggregate
    // was the only robust guard because rotated-eigenvector cases regressed;
    // the preconditioned surrogate converges to the exact dip constraint, so
    // the aggregate must now strictly improve.
    let mut plain_total = 0.0_f32;
    let mut refined_total = 0.0_f32;

    for distortion in distortions {
        for (world_mag, world_gravity) in dip_cases {
            let mut plain = MagCalibrator::<63>::new();
            // The default weight keeps the surrogate active; this sweep
            // characterizes its convergence under anisotropy.
            let mut gravity_refined = MagCalibrator::<63>::new();
            for i in 0..16 * 63 {
                let j = i % 63;
                let attitude = UnitQuaternion::from_euler_angles(
                    0.25 * (j as f32 * 0.7).sin(),
                    0.35 * (j as f32 * 1.7).sin(),
                    j as f32 * 2.4,
                );
                let body_mag = attitude.inverse() * world_mag;
                let body_gravity = attitude.inverse() * world_gravity;
                let raw = offset + distortion * body_mag;
                let _ = plain.evaluate_correct(raw, None, i as u64);
                let _ = gravity_refined.evaluate_correct(raw, Some(body_gravity), i as u64);
            }

            let probes = [
                Vector3::x(),
                Vector3::y(),
                Vector3::z(),
                Vector3::new(1.0, -2.0, 3.0).normalize(),
            ];
            let error_of = |calibrator: &MagCalibrator<63>| {
                probes
                    .iter()
                    .map(|&expected| {
                        (calibrator
                            .correct_working_for_test(offset + distortion * expected)
                            .expect("working calibration is invalid")
                            - expected)
                            .norm()
                    })
                    .sum::<f32>()
            };
            let plain_error = error_of(&plain);
            let refined_error = error_of(&gravity_refined);
            // The preconditioned surrogate converges to the exact dip, so
            // every anisotropy/dip case improves individually, not just on
            // aggregate.
            assert!(
                refined_error < plain_error,
                "case regressed: plain={plain_error} refined={refined_error}"
            );
            plain_total += plain_error;
            refined_total += refined_error;
        }
    }

    // The unpreconditioned surrogate pinned `g_i^T A m_i` rather than the
    // exact dip `g_i^T m_i`, so rotated full-SPD soft iron biased the fit
    // toward isotropy: the rotated cases above regressed at every tested
    // nonzero weight (already +0.3 aggregate probe error at weight `0.003`).
    // The preconditioned surrogate pins `g_i^T A_w^{-1} A m_i`, which
    // converges to the exact dip as the working correction converges; the
    // regression is gone and every case in this sweep improves instead.
    // The aggregate must therefore be strictly better than the plain fit —
    // a tolerance here would let an anisotropy bias creep back in.
    assert!(
        refined_total < plain_total,
        "preconditioned gravity surrogate must improve under strong anisotropy: \
         plain_total={plain_total} refined_total={refined_total}"
    );
}

#[test]
fn mag_calibrator_uses_gravity_by_default() {
    // With the preconditioned surrogate, gravity hints inform the fit at the
    // default weight: a default calibrator fed consistent gravity must keep
    // a live gravity statistic and converge differently from a hint-free
    // one. An explicit gravity_weight(0) must however restore the old
    // disabled behavior exactly: the valid hints then change nothing, not
    // even a cached-row side effect.
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    // Co-rotating world directions keep the dip angle constant, so the hint
    // stream is physically consistent with the magnetic samples.
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
    let mut plain = MagCalibrator::<63>::new();
    let mut hinted = MagCalibrator::<63>::new();
    let mut opted_out = MagCalibrator::<63>::new().gravity_weight(0.0);
    let mut hinted_result = None;
    for i in 0..17 * 63 {
        let j = i % 63;
        let attitude = UnitQuaternion::from_euler_angles(
            0.25 * (j as f32 * 0.7).sin(),
            0.35 * (j as f32 * 1.7).sin(),
            j as f32 * 2.4,
        );
        let body_gravity = attitude.inverse() * world_gravity;
        let raw = offset + distortion * (attitude.inverse() * world_mag);
        assert_eq!(
            plain.evaluate_correct(raw, None, i as u64),
            opted_out.evaluate_correct(raw, Some(body_gravity), i as u64)
        );
        hinted_result = Some(hinted.evaluate_correct(raw, Some(body_gravity), i as u64));
    }
    let hinted_result = hinted_result.unwrap().unwrap();
    // The gravity term is live and, with a consistent hint stream,
    // converged: the loss sits at or below the converged bound — the
    // loss-domain equivalent of the former fitness plateau, whose RMS
    // residual floor 0.1 maps to a mean square of 0.01 and a loss of
    // `0.5 * 0.01 * 0.01 = 5.0e-5` at the default weight. It also shifts
    // the working fit, so the hinted run cannot remain bit-identical to
    // the hint-free one.
    assert!(
        hinted_result.gravity_loss <= 5.0e-5,
        "gravity_loss={}",
        hinted_result.gravity_loss
    );
    assert!(
        hinted.model.parameters != plain.model.parameters,
        "default gravity surrogate left the fit untouched"
    );

    // The learned dip projection converges to the exact dip: with the
    // co-rotated hint stream the reported `dip_sin` is the constant
    // `g^T m` of the world pair, while a gravity-disabled calibrator never
    // reports one.
    let expected_dip_sin = world_gravity.dot(&world_mag);
    let dip_sin = hinted_result
        .dip_sin
        .expect("hinted calibrator must report a dip projection");
    assert!(
        (dip_sin - expected_dip_sin).abs() < 0.05,
        "dip_sin={dip_sin}, expected={expected_dip_sin}"
    );

    // No gravity term exists when gravity is disabled: a hinted-but-disabled
    // calibrator has no seed and no residual scan.
    assert_eq!(opted_out.model.learned_gravity_projection, None);
    let opted_out_result = opted_out
        .evaluate_correct(offset + distortion * sample_direction(0, 63), None, 17 * 63)
        .unwrap();
    assert_eq!(opted_out_result.gravity_loss, 0.0);
    assert_eq!(opted_out_result.dip_sin, None);
}

/// One independently rederived online update against production: both the
/// sample-anchored and the cache-replay form. The test duplicates the draw
/// sequence, the feature construction, the gradient accumulation, and the
/// diagonal scaling from the documented formulas, then asserts that the
/// production update moves the working state along the negative normalized
/// subgradient of exactly that minibatch and strictly lowers its objective.
/// With a live projection every draw carries gravity, so the drawn gravity
/// subset equals the whole minibatch and both gradient terms are exercised.
#[track_caller]
fn check_minibatch_update_against_analytic_subgradient(
    calibrator: &mut MagCalibrator<63>,
    current: Option<(Vector3<f32>, Vector3<f32>)>,
    random_draws: usize,
) {
    let parameters = calibrator.model.parameters;
    let kappa = calibrator
        .model
        .learned_gravity_projection
        .expect("gravity projection must be seeded");
    let weight = calibrator.model.gravity_weight;
    let spec = MinibatchSpec {
        current_sample: current.map(|(sample, _)| sample),
        current_gravity: current.map(|(_, gravity)| gravity),
        accepted_row: None,
        random_draws,
        random_state: calibrator.prng_state,
    };

    // Duplicate the draw sequence and feature construction of
    // `minibatch_feature_matrices`. The frame and normalization are frozen
    // for the whole update, so the cached rows are read once up front.
    let mut random_state = spec.random_state;
    let mut observations: Vec<(Vector3<f32>, Option<Vector3<f32>>)> = current
        .map(|(sample, gravity)| (sample, Some(gravity)))
        .into_iter()
        .collect();
    for _ in 0..random_draws {
        let row = MagCalibrator::<63>::random_cache_row(
            &mut random_state,
            calibrator.model.samples.stats.sample_row_count,
            None,
        )
        .expect("retained cache is empty");
        let row = calibrator.model.samples.view(row);
        observations.push((row.sample(), row.gravity()));
    }
    let radial_features: Vec<_> = observations
        .iter()
        .map(|&(sample, _)| {
            MagModel::<63>::features(calibrator.model.samples.stats.normalized_sample(sample))
        })
        .collect();
    let gravity_features: Vec<_> = observations
        .iter()
        .filter_map(|&(sample, gravity)| {
            gravity.map(|gravity| {
                MagModel::<63>::gravity_features(
                    calibrator.model.samples.stats.normalized_sample(sample),
                    calibrator.model.preconditioned_gravity(gravity),
                )
            })
        })
        .collect();
    assert_eq!(
        gravity_features.len(),
        observations.len(),
        "every warmup row carries gravity, so the gravity subset is the whole minibatch"
    );
    let observation_count = observations.len() as f32;
    let gravity_count = gravity_features.len() as f32;

    // Analytic gradient and diagonal scales of the documented objective.
    // The gravity projection scale is frozen from the pre-update parameters,
    // exactly as production freezes it for the whole update.
    let projections: Vec<f32> = gravity_features
        .iter()
        .map(|features| features.dot(&parameters))
        .collect();
    let gravity_scale = (projections.iter().map(|p| p * p).sum::<f32>() / gravity_count)
        .max(ONLINE_SCALE_EPSILON)
        .sqrt();
    let gravity_scale_squared = gravity_scale * gravity_scale;
    let prior = MagModel::<63>::parameter_prior();
    let regularization_weights = SVector::<f32, CALIBRATION_PARAMETER_COUNT>::from_row_slice(&[
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
    ]);
    let mut gradient = SVector::<f32, CALIBRATION_PARAMETER_COUNT>::zeros();
    let mut gradient_scale = SVector::<f32, CALIBRATION_PARAMETER_COUNT>::zeros();
    for features in &radial_features {
        gradient += features * (features.dot(&parameters) - 1.0);
        gradient_scale += features.component_mul(features);
    }
    gradient /= observation_count;
    gradient_scale /= observation_count;
    let mut kappa_gradient = 0.0f32;
    for features in &gravity_features {
        let residual = features.dot(&parameters) - kappa;
        gradient += weight / gravity_scale_squared / gravity_count * features * residual;
        gradient_scale +=
            weight / gravity_scale_squared / gravity_count * features.component_mul(features);
        kappa_gradient -= residual;
    }
    kappa_gradient *= weight / gravity_scale_squared / gravity_count;
    gradient += SHAPE_REGULARIZATION * regularization_weights.component_mul(&(parameters - prior));
    gradient_scale += SHAPE_REGULARIZATION * regularization_weights;
    gradient_scale.add_scalar_mut(ONLINE_SCALE_EPSILON);
    let descent = gradient.component_div(&gradient_scale);
    let kappa_step = kappa_gradient / (weight / gravity_scale_squared + ONLINE_SCALE_EPSILON);

    // The same minibatch objective as production, for the decrease check:
    // the gravity residual is relative to the frozen projection scale.
    let objective = |parameters: &SVector<f32, CALIBRATION_PARAMETER_COUNT>, kappa: f32| {
        let radial: f32 = radial_features
            .iter()
            .map(|features| {
                let residual = features.dot(parameters) - 1.0;
                residual * residual
            })
            .sum();
        let gravity: f32 = gravity_features
            .iter()
            .map(|features| {
                let residual = (features.dot(parameters) - kappa) / gravity_scale;
                residual * residual
            })
            .sum();
        0.5 * radial / observation_count
            + 0.5 * weight * gravity / gravity_count
            + 0.5
                * SHAPE_REGULARIZATION
                * regularization_weights
                    .component_mul(&(parameters - prior))
                    .dot(&(parameters - prior))
    };
    let objective_before = objective(&parameters, kappa);

    assert!(
        calibrator.apply_minibatch_update(spec),
        "a gravity-carrying minibatch with a live gradient was rejected"
    );
    let delta_theta = calibrator.model.parameters - parameters;
    let delta_kappa = calibrator.model.learned_gravity_projection.unwrap() - kappa;

    // The accepted step must be exactly anti-parallel to the normalized
    // subgradient: any wrongly sampled row, feature, weighting, or sign in
    // the production gradient rotates the accepted direction.
    let moved = (delta_theta.norm_squared() + delta_kappa * delta_kappa).sqrt();
    let expected_norm = (descent.norm_squared() + kappa_step * kappa_step).sqrt();
    assert!(moved > 0.0 && expected_norm > 0.0);
    let alignment =
        -(delta_theta.dot(&descent) + delta_kappa * kappa_step) / (moved * expected_norm);
    assert!(
        (alignment - 1.0).abs() < 1.0e-4,
        "accepted step deviates from the analytic subgradient: alignment={alignment}"
    );

    let objective_after = objective(&calibrator.model.parameters, kappa + delta_kappa);
    assert!(
        objective_after < objective_before,
        "accepted step did not lower the minibatch objective: \
         before={objective_before} after={objective_after}"
    );
}

#[test]
fn mag_calibrator_gravity_subgradient_matches_online_update() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
    // Replay disabled so the warmup performs exactly one optimizer update
    // per sample and this test observes isolated later updates.
    let mut calibrator = MagCalibrator::<63>::new().replay_updates(0);
    for i in 0..63 {
        let attitude = UnitQuaternion::from_euler_angles(
            0.25 * (i as f32 * 0.7).sin(),
            0.35 * (i as f32 * 1.7).sin(),
            i as f32 * 2.4,
        );
        let raw = offset + distortion * (attitude.inverse() * world_mag);
        let body_gravity = attitude.inverse() * world_gravity;
        calibrator.evaluate_sample_vec(raw, Some(body_gravity), i as u64);
    }
    assert!(calibrator.model.learned_gravity_projection.is_some());

    // Sample-anchored update: the arriving observation plus gravity-carrying
    // cache draws.
    let next_sample = offset + distortion * sample_direction(7, 63);
    let next_gravity = Vector3::z();
    check_minibatch_update_against_analytic_subgradient(
        &mut calibrator,
        Some((next_sample, next_gravity)),
        31,
    );
    // Cache-replay update: no anchoring observation, retained rows only.
    check_minibatch_update_against_analytic_subgradient(&mut calibrator, None, 8);
}

#[test]
fn mag_calibrator_ignores_invalid_gravity() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut plain = MagCalibrator::<12>::new();
    // The default weight keeps the gravity term active, so this exercises
    // invalid-gravity filtering with a live surrogate.
    let mut invalid = MagCalibrator::<12>::new();
    for i in 0..12 {
        let raw = offset + distortion * sample_direction(i, 12);
        let _ = plain.evaluate_correct(raw, None, i as u64);
        let gravity = match i % 3 {
            0 => Vector3::repeat(f32::NAN),
            1 => Vector3::repeat(f32::MAX),
            _ => Vector3::zeros(),
        };
        let _ = invalid.evaluate_correct(raw, Some(gravity), i as u64);
    }
    let training_updates = 2 * MIN_PUBLICATION_STREAK;
    for i in 12..12 + training_updates {
        let raw = offset + distortion * sample_direction(i % 12, 12);
        let _ = plain.evaluate_correct(raw, None, i as u64);
        let gravity = match i % 3 {
            0 => Vector3::repeat(f32::NAN),
            1 => Vector3::repeat(f32::MAX),
            _ => Vector3::zeros(),
        };
        let _ = invalid.evaluate_correct(raw, Some(gravity), i as u64);
    }

    let expected = Vector3::new(1.0, -2.0, 3.0).normalize();
    let raw = offset + distortion * expected;
    let plain = calibrated(plain.evaluate_correct(raw, None, 100));
    let invalid = calibrated(invalid.evaluate_correct(raw, None, 100));
    assert_vec_close(invalid, plain, 1.0e-6);
}

#[test]
fn mag_calibrator_scores_direction_coverage() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(2.4, 0.3, -0.2, 0.3, 0.7, 0.1, -0.2, 0.1, 1.3);
    let mut calibrator = MagCalibrator::<63>::new();

    // A near-planar circle leaves the directional design matrix
    // rank-deficient no matter how long it is sampled, and the coverage
    // score cannot be inflated by the fit's own reshaping of the sample
    // covariance.
    for i in 0..16 * 63 {
        let azimuth = i as f32 * 0.31;
        let near_planar =
            Vector3::new(azimuth.cos() * 0.866, azimuth.sin() * 0.866, 0.5).normalize();
        calibrator.evaluate_sample_vec(offset + near_planar, None, i as u64);
    }
    let (valid, coverage, _, _) = calibrator.working_quality_components();
    assert!(valid, "planar sweep produced no working candidate");
    assert!(
        coverage < MIN_PUBLICATION_CONFIDENCE,
        "planar coverage={coverage} must stay below the publication confidence"
    );
    assert!(calibrator.get_confidence() < MIN_PUBLICATION_CONFIDENCE);

    // The same broad three-dimensional sweep under identity and strong
    // anisotropic distortion must score similarly: coverage follows the
    // retained directions, not the fitted correction.
    let mut identity = MagCalibrator::<63>::new();
    let mut distorted = MagCalibrator::<63>::new();
    for i in 0..63 {
        let direction = sample_direction(i, 63);
        identity.evaluate_sample_vec(offset + direction, None, i as u64);
        distorted.evaluate_sample_vec(offset + distortion * direction, None, i as u64);
    }
    let (identity_valid, identity_coverage, _, _) = identity.working_quality_components();
    let (distorted_valid, distorted_coverage, _, _) = distorted.working_quality_components();
    assert!(identity_valid && distorted_valid);
    assert!(
        (identity_coverage - distorted_coverage).abs() < 0.2,
        "identity_coverage={identity_coverage} distorted_coverage={distorted_coverage}"
    );
    assert!(
        distorted_coverage >= MIN_PUBLICATION_CONFIDENCE,
        "identity_coverage={identity_coverage} distorted_coverage={distorted_coverage} \
         threshold={MIN_PUBLICATION_CONFIDENCE}"
    );
}

#[test]
fn design_coverage_is_rotation_invariant_and_detects_rank_deficiency() {
    // A broad deterministic sweep has near-isotropic directional support and
    // scores close to the uniform-sphere reference.
    let directions: Vec<Vector3<f32>> = (0..64).map(|i| sample_direction(i, 64)).collect();
    let coverage = MagCalibrator::<9>::coverage_scores_for_test(
        &MagCalibrator::<9>::coverage_gram_sum_for_test(&directions),
        64,
    );
    // The spiral never visits the poles, so it scores well below the
    // uniform-sphere reference but far above a rank-deficient sweep.
    assert!(coverage > 0.3, "broad coverage={coverage}");
    assert!(coverage <= 1.0, "coverage={coverage} exceeds the clamp");

    // The sqrt(2)-weighted features make the induced rotation on feature
    // space orthogonal, so a rigid rotation of every direction leaves the
    // score unchanged.
    let rotation = UnitQuaternion::from_euler_angles(0.4, -0.7, 1.1);
    let rotated: Vec<Vector3<f32>> = directions.iter().map(|&d| rotation * d).collect();
    let rotated_coverage = MagCalibrator::<9>::coverage_scores_for_test(
        &MagCalibrator::<9>::coverage_gram_sum_for_test(&rotated),
        64,
    );
    assert!(
        (coverage - rotated_coverage).abs() < 1.0e-4,
        "coverage={coverage} rotated_coverage={rotated_coverage}"
    );

    // A tilted circle spans a measure-zero band: the design matrix is
    // rank-deficient and the score collapses however long the circle runs.
    let circle: Vec<Vector3<f32>> = (0..64)
        .map(|i| {
            let azimuth = i as f32 * 0.31;
            Vector3::new(azimuth.cos() * 0.866, azimuth.sin() * 0.866, 0.5).normalize()
        })
        .collect();
    let planar_coverage = MagCalibrator::<9>::coverage_scores_for_test(
        &MagCalibrator::<9>::coverage_gram_sum_for_test(&circle),
        64,
    );
    assert!(
        planar_coverage < 0.1,
        "planar_coverage={planar_coverage} must stay near zero"
    );

    // Fewer retained rows than the nine fit features score zero.
    assert_eq!(
        MagCalibrator::<9>::coverage_scores_for_test(
            &MagCalibrator::<9>::coverage_gram_sum_for_test(&directions[..8]),
            8,
        ),
        0.0
    );
}

#[test]
fn mag_calibrator_reports_confidence_factors() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);

    // Without gravity the gravity factor stays neutral, and confidence is
    // the coverage factor alone.
    let mut plain = MagCalibrator::<63>::new();
    let mut plain_result = None;
    for i in 0..16 * 63 {
        let raw = offset + distortion * sample_direction(i % 63, 63);
        plain_result = Some(plain.evaluate_correct(raw, None, i as u64).unwrap());
    }
    let plain = plain_result.unwrap();
    // Without gravity the objective contains no gravity term, and
    // confidence is the coverage factor alone.
    assert_eq!(plain.gravity_loss, 0.0);
    assert_eq!(plain.confidence(), plain.coverage.clamp(0.0, 1.0));

    // With a consistent co-rotating gravity direction the gravity term is
    // live. Gravity fixed in the body frame while the attitude rotates is
    // physically contradictory: no constant dip angle exists, the
    // projection residual stays large, and the loss rises.
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
    // The default weight keeps the surrogate active in both calibrators.
    let mut refined = MagCalibrator::<63>::new();
    let mut opposed = MagCalibrator::<63>::new();
    let mut refined_result = None;
    let mut opposed_result = None;
    for i in 0..16 * 63 {
        let j = i % 63;
        let attitude = UnitQuaternion::from_euler_angles(
            0.25 * (j as f32 * 0.7).sin(),
            0.35 * (j as f32 * 1.7).sin(),
            j as f32 * 2.4,
        );
        let body_mag = attitude.inverse() * world_mag;
        let body_gravity = attitude.inverse() * world_gravity;
        let raw = offset + distortion * body_mag;
        refined_result = Some(
            refined
                .evaluate_correct(raw, Some(body_gravity), i as u64)
                .unwrap(),
        );
        opposed_result = Some(
            opposed
                .evaluate_correct(raw, Some(Vector3::z()), i as u64)
                .unwrap(),
        );
    }
    let refined = refined_result.unwrap();
    let opposed = opposed_result.unwrap();
    // With the preconditioned surrogate, a consistent co-rotating gravity
    // direction is fit exactly once the frame converges: the projection
    // residual collapses to the converged bound — at or below `5.0e-5`,
    // the loss-domain equivalent of the former fitness plateau (an RMS
    // residual at or below 0.1, i.e. a mean square at or below 0.01, at
    // the default weight 0.01).
    assert!(
        refined.gravity_loss <= 5.0e-5,
        "gravity_loss={}",
        refined.gravity_loss
    );
    // Confidence is the clamped coverage factor alone: the loss
    // diagnostics are reported but take no part in it.
    assert_eq!(refined.confidence(), refined.coverage.clamp(0.0, 1.0));
    assert!(
        opposed.gravity_loss > refined.gravity_loss,
        "opposed={} refined={}",
        opposed.gravity_loss,
        refined.gravity_loss
    );
}

#[test]
fn raw_moments_follow_all_cache_mutations_and_const_generic_edges() {
    let mut calibrator = MagCalibrator::<3>::new()
        .num_neighbors(2)
        .max_sample_lifespan_us(5);
    let samples = [Vector3::x(), Vector3::y(), Vector3::z()];
    for (timestamp_us, sample) in samples.into_iter().enumerate() {
        calibrator.evaluate_sample_vec(sample, None, timestamp_us as u64);
        calibrator.check_raw_moments().unwrap();
    }

    let full_moments = calibrator.raw_moments_for_test();
    calibrator.evaluate_sample_vec(Vector3::x(), None, 3);
    assert_eq!(calibrator.raw_moments_for_test(), full_moments);
    calibrator.check_raw_moments().unwrap();

    calibrator.evaluate_sample_vec(Vector3::repeat(10.0), None, 4);
    assert_ne!(calibrator.raw_moments_for_test(), full_moments);
    calibrator.check_raw_moments().unwrap();

    calibrator.evaluate_sample_vec(Vector3::repeat(f32::NAN), None, 7);
    assert_eq!(calibrator.raw_moments_for_test().0, 2);
    calibrator.check_raw_moments().unwrap();

    calibrator.evaluate_sample_vec(Vector3::repeat(f32::INFINITY), None, 10);
    assert_eq!(
        calibrator.raw_moments_for_test(),
        (0, Vector3::zeros(), Matrix3::zeros())
    );
    calibrator.check_raw_moments().unwrap();

    let mut empty = MagCalibrator::<0>::new();
    empty.evaluate_sample_vec(Vector3::x(), None, 0);
    empty.evaluate_sample_vec(Vector3::repeat(f32::NAN), None, 1);
    assert_eq!(
        empty.raw_moments_for_test(),
        (0, Vector3::zeros(), Matrix3::zeros())
    );
    empty.check_raw_moments().unwrap();

    let mut singleton = MagCalibrator::<1>::new().max_sample_lifespan_us(0);
    singleton.evaluate_sample_vec(Vector3::x(), None, 0);
    let singleton_moments = singleton.raw_moments_for_test();
    singleton.evaluate_sample_vec(Vector3::y(), None, 0);
    assert_eq!(singleton.raw_moments_for_test(), singleton_moments);
    singleton.evaluate_sample_vec(Vector3::repeat(f32::NAN), None, 1);
    assert_eq!(
        singleton.raw_moments_for_test(),
        (0, Vector3::zeros(), Matrix3::zeros())
    );
    singleton.check_raw_moments().unwrap();
}

#[test]
fn mag_calibrator_neighbor_cache_matches_naive_rescan() {
    // Deterministic xorshift64 PRNG.
    let mut prng_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        prng_state ^= prng_state << 13;
        prng_state ^= prng_state >> 7;
        prng_state ^= prng_state << 17;
        prng_state
    };

    // N - 1 = 11 exceeds the neighbor cache capacity of 8, so caches are
    // incomplete and inserts are dropped once the pad is exhausted; k = 10
    // additionally exercises the above-capacity direct-scan fallback. The
    // short lifespan keeps expiry compaction and buffer refills in the mix.
    for k in [2, 3, 10] {
        let mut calibrator = MagCalibrator::<12>::new()
            .num_neighbors(k)
            .max_sample_lifespan_us(25);
        let mut expected_rows: Vec<(Vector3<f32>, u64)> = Vec::new();
        let mut timestamp_us = 0;
        for _ in 0..4000 {
            timestamp_us += 1 + next() % 3;
            // Quantized directions with jitter: new samples frequently land
            // near buffered ones, provoking replacements.
            let azimuth = (next() % 8) as f32 * 0.785 + (next() % 100) as f32 / 500.0;
            let z = (next() % 5) as f32 / 2.5 - 1.0 + (next() % 100) as f32 / 500.0;
            let xy_radius = (1.0 - z * z).max(0.0).sqrt();
            let direction = Vector3::new(xy_radius * azimuth.cos(), xy_radius * azimuth.sin(), z);
            let sample = Vector3::new(11.0, -7.0, 5.0) + 40.0 * direction;
            expected_rows.retain(|(_, time)| timestamp_us.saturating_sub(*time) <= 25);
            if expected_rows.len() < 12 {
                expected_rows.push((sample, timestamp_us));
            } else {
                let mean_distance = |point: Vector3<f32>, excluded: usize| {
                    let mut distances: Vec<f32> = expected_rows
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != excluded)
                        .map(|(_, (other, _))| (point - other).norm())
                        .collect();
                    distances.sort_unstable_by(f32::total_cmp);
                    distances[..k].iter().sum::<f32>() / k as f32
                };
                let (victim, score) = expected_rows
                    .iter()
                    .enumerate()
                    .map(|(index, (point, _))| (index, mean_distance(*point, index)))
                    .min_by(|(_, left), (_, right)| left.total_cmp(right))
                    .unwrap();
                if mean_distance(sample, victim) > score {
                    expected_rows[victim] = (sample, timestamp_us);
                }
            }
            calibrator.evaluate_sample_vec(sample, None, timestamp_us);
            assert_eq!(
                calibrator.model.samples.stats.sample_row_count,
                expected_rows.len()
            );
            for (index, (expected_sample, expected_time)) in expected_rows.iter().enumerate() {
                assert_eq!(
                    calibrator.model.samples.view(index).sample(),
                    *expected_sample
                );
                assert_eq!(
                    calibrator.model.samples.timestamps_us[index],
                    *expected_time
                );
            }
            calibrator
                .check_neighbor_cache()
                .unwrap_or_else(|message| panic!("k={k} timestamp_us={timestamp_us}: {message}"));
            calibrator
                .check_raw_moments()
                .unwrap_or_else(|message| panic!("k={k} timestamp_us={timestamp_us}: {message}"));
        }
    }
}

#[test]
fn mag_calibrator_candidate_score_includes_replaced_victim() {
    // Cluster of three near-duplicate rows around `victim` (distance 0.01),
    // with nine well-separated rows far away. The victim is the unique row
    // with the lowest k=2 mean nearest distance.
    let mut calibrator = MagCalibrator::<12>::new().num_neighbors(2);
    let victim = Vector3::new(5.0, 5.0, 5.0);
    let cluster = [
        victim,
        Vector3::new(4.99, 5.0, 5.0),
        Vector3::new(5.0, 4.99, 5.0),
    ];
    let mut points = cluster.to_vec();
    for i in 0..9 {
        points.push(Vector3::new(6.0 + 0.5 * i as f32, 5.0, 5.0));
    }
    for (timestamp_us, sample) in points.iter().copied().enumerate() {
        calibrator.evaluate_sample_vec(sample, None, timestamp_us as u64);
    }

    // Identify the victim exactly as the replacement branch does, and build a
    // candidate adjacent to it but far from its cluster companions and every
    // other row. Excluding the victim, its k=2 nearest neighbors become the
    // two cluster companions (~0.011 and ~0.01005), scoring above the
    // victim's own ~0.01; including the victim (~0.001), its score drops to
    // ~0.0055, below the victim's.
    let (replacement_row, _) = calibrator.lowest_mean_distance_by_index();
    assert_eq!(
        calibrator.model.samples.view(replacement_row).sample(),
        victim
    );
    let candidate = Vector3::new(5.001, 5.0, 5.0);

    calibrator.evaluate_sample_vec(candidate, None, 12);

    // Correct post-replacement behavior: the victim is evicted and the
    // candidate takes its row. This assertion fails while the candidate is
    // still scored against the victim row it would replace.
    assert_eq!(
        calibrator.model.samples.view(replacement_row).sample(),
        candidate,
        "candidate adjacent to the victim was wrongly rejected"
    );
    calibrator.check_neighbor_cache().unwrap();
    calibrator.check_raw_moments().unwrap();
}

fn seeded_calibrator<const N: usize>(
    offset: Vector3<f32>,
    distortion: Matrix3<f32>,
) -> MagCalibrator<N> {
    train_calibrator(MagCalibrator::new(), offset, distortion)
}

fn train_calibrator<const N: usize>(
    mut calibrator: MagCalibrator<N>,
    offset: Vector3<f32>,
    distortion: Matrix3<f32>,
) -> MagCalibrator<N> {
    for i in 0..N {
        let direction = sample_direction(i, N);
        let _ = calibrator.evaluate_correct(offset + distortion * direction, None, 0);
    }
    let mut result = None;
    let training_updates = (16 * N).max(2 * MIN_PUBLICATION_STREAK);
    for i in 0..training_updates {
        let direction = sample_direction(i % N, N);
        result = Some(calibrator.evaluate_correct(offset + distortion * direction, None, 0));
    }
    assert!(
        result.is_some_and(|result| result.is_ok_and(|result| result.direction.is_some())),
        "online calibration did not converge"
    );
    calibrator
}

fn sample_direction(i: usize, n: usize) -> Vector3<f32> {
    let azimuth = 0.37 + i as f32 * 1.21;
    let z = -0.8 + 1.6 * i as f32 / (n - 1) as f32;
    let xy_radius = (1.0 - z * z).sqrt();
    Vector3::new(xy_radius * azimuth.cos(), xy_radius * azimuth.sin(), z)
}

fn assert_vec_close(actual: Vector3<f32>, expected: Vector3<f32>, tolerance: f32) {
    let diff = (actual - expected).norm();
    assert!(
        diff < tolerance,
        "actual={} expected={} diff={}",
        actual.transpose(),
        expected.transpose(),
        diff
    );
}

fn calibrated(result: Result<MagCalibrationResult, BadMagCause>) -> Vector3<f32> {
    result
        .expect("magnetometer evaluation failed")
        .direction
        .expect("calibration is still pending")
}
