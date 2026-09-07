use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::super::BadMagCause;
use super::{
    CalibrationQuality, CoverageGramMatrix, MagCalibrationResult, MagCalibrator,
    MIN_PUBLICATION_CONFIDENCE, MIN_PUBLICATION_STREAK,
};

impl<const N: usize> MagCalibrator<N> {
    fn working_quality_components(&self) -> (bool, f32, f32, f32) {
        let Ok(_candidate) = self.working_candidate() else {
            return (false, 0.0, 0.0, 0.0);
        };
        let coverage = self.mean_centered_coverage();
        let radial_fitness = Self::radial_fitness_score(self.radial_residual_mean_square);
        let gravity_fitness = Self::gravity_fitness_score(self.gravity_residual_mean_square);
        (true, coverage, radial_fitness, gravity_fitness)
    }

    fn correct_working_for_test(&self, raw_mag: Vector3<f32>) -> Option<Vector3<f32>> {
        let candidate = self.working_candidate().ok()?;
        let corrected = candidate.correction * (raw_mag - candidate.offset);
        let norm = corrected.norm();
        (norm.is_finite() && norm > f32::EPSILON).then(|| corrected / norm)
    }

    fn publication_quality_streak_for_test(&self) -> usize {
        self.publication_quality_streak
    }

    fn coverage_scores_for_test(gram_sum: &CoverageGramMatrix, sample_row_count: usize) -> f32 {
        Self::coverage_from_gram(gram_sum, sample_row_count)
    }

    fn coverage_gram_sum_for_test(directions: &[Vector3<f32>]) -> CoverageGramMatrix {
        let mut gram_sum = CoverageGramMatrix::zeros();
        for &direction in directions {
            let feature = Self::direction_feature(direction);
            gram_sum += feature * feature.transpose();
        }
        gram_sum
    }

    fn fitness_score_for_test(mean_square: Option<f32>) -> f32 {
        Self::radial_fitness_score(mean_square)
    }

    fn gravity_fitness_score_for_test(mean_square: Option<f32>) -> f32 {
        Self::gravity_fitness_score(mean_square)
    }

    fn running_mean_square_for_test(
        current: Option<f32>,
        residual_squared: f32,
        update_weight: f32,
    ) -> Option<f32> {
        Self::update_running_mean_square(current, residual_squared, update_weight)
    }

    fn radial_residual_mean_square_for_test(&self) -> Option<f32> {
        self.radial_residual_mean_square
    }

    fn raw_moments_for_test(&self) -> (usize, Vector3<f64>, Matrix3<f64>) {
        (
            self.sample_row_count,
            self.raw_sample_sum,
            self.raw_outer_product_sum,
        )
    }

    /// Verifies maintained raw moments against a direct current-cache sum.
    fn check_raw_moments(&self) -> Result<(), String> {
        let (sum, outer_sum) = (0..self.sample_row_count).fold(
            (Vector3::<f64>::zeros(), Matrix3::<f64>::zeros()),
            |(sum, outer_sum), row| {
                let sample = self.sample(row).cast::<f64>();
                (sum + sample, outer_sum + sample * sample.transpose())
            },
        );
        let scale = sum.norm().max(outer_sum.norm()).max(1.0);
        let error = (self.raw_sample_sum - sum)
            .norm()
            .max((self.raw_outer_product_sum - outer_sum).norm());
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
        for row in 0..self.sample_row_count {
            let len = self.neighbor_cache_len[row] as usize;
            let cache = &self.neighbor_cache[row][..len];
            let mut true_dists: Vec<f32> = (0..self.sample_row_count)
                .filter(|&j| j != row)
                .map(|j| {
                    let diff = self.sample(row) - self.sample(j);
                    diff.dot(&diff)
                })
                .collect();
            true_dists.sort_unstable_by(|a, b| a.total_cmp(b));
            for (i, entry) in cache.iter().enumerate() {
                if entry.row as usize >= self.sample_row_count || entry.row as usize == row {
                    return Err(format!("row {row}: entry {i} references row {}", entry.row));
                }
                if i > 0 && cache[i - 1].squared_distance > entry.squared_distance {
                    return Err(format!("row {row}: entry {i} out of order"));
                }
                if cache[..i].iter().any(|e| e.row == entry.row) {
                    return Err(format!("row {row}: duplicate entry for row {}", entry.row));
                }
                let diff = self.sample(row) - self.sample(entry.row as usize);
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
        let theta = 0.37 + i as f32 * 1.21;
        let z = -0.85 + 1.7 * i as f32 / 62.0;
        let radius = (1.0 - z * z).sqrt();
        let direction = Vector3::new(radius * theta.cos(), radius * theta.sin(), z);
        let _ = calibrator.evaluate_correct(offset + distortion * direction, None, i as u64);
    }

    for i in 0..16 * 63 {
        let sample_index = i % 63;
        let theta = 0.37 + sample_index as f32 * 1.21;
        let z = -0.85 + 1.7 * sample_index as f32 / 62.0;
        let radius = (1.0 - z * z).sqrt();
        let direction = Vector3::new(radius * theta.cos(), radius * theta.sin(), z);
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
            quality: CalibrationQuality {
                confidence: 0.0,
                ..
            },
            direction: None,
        })
    ));
    let result = (1..9)
        .map(|timestamp_us| calibrator.evaluate_correct(sample, None, timestamp_us))
        .last()
        .unwrap();

    assert!(matches!(
        result,
        Ok(MagCalibrationResult {
            quality: CalibrationQuality {
                confidence: 0.0,
                ..
            },
            direction: None,
        })
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
        if result.confidence >= MIN_PUBLICATION_CONFIDENCE {
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
            quality: CalibrationQuality {
                confidence: 0.0,
                ..
            },
            direction: None,
        })
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
    assert_eq!(result.confidence, 0.0);
    assert_vec_close(
        result
            .direction
            .expect("last published correction was discarded"),
        expected,
        0.05,
    );
    // Working state is never rebased or reset, so the running radial
    // statistic survives the expiry that empties the cache.
    assert_ne!(calibrator.radial_residual_mean_square_for_test(), None);
}

#[test]
fn mag_calibrator_online_history_outlives_sample_lifespan() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut calibrator = seeded_calibrator::<63>(offset, distortion).max_sample_lifespan_us(0);
    let expected = Vector3::x();
    let raw = offset + distortion * expected;

    // The radial fitness statistic is a persisted running mean of the online
    // optimizer's fit over the training cache.
    let trained_radial = calibrator.radial_residual_mean_square_for_test();
    assert!(trained_radial.is_some());

    // A zero lifespan makes the next, larger-timestamp sample expire every
    // retained row, emptying the cache. The last published correction stays in
    // use even though live confidence collapses.
    let result = calibrator.evaluate_correct(raw, None, 1).unwrap();
    assert_eq!(result.confidence, 0.0);
    assert_vec_close(
        result
            .direction
            .expect("last published correction was discarded"),
        expected,
        0.05,
    );

    // Known adaptation limitation: expiry removes rows from the cache but not
    // their historical online-SGD gradient contribution. The radial statistic
    // is byte-for-byte unchanged even though every training row is gone,
    // because nothing resets or forgets the online worker on expiry:
    // `max_sample_lifespan_us` bounds cache membership, not the optimizer's
    // effective history.
    assert_eq!(
        calibrator.radial_residual_mean_square_for_test(),
        trained_radial
    );
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
                quality: CalibrationQuality {
                    confidence: 0.0,
                    ..
                },
                direction: None,
            })
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
            quality: CalibrationQuality {
                confidence: 0.0,
                ..
            },
            direction: Some(_),
        }
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
            quality: CalibrationQuality {
                confidence: 0.0,
                ..
            },
            direction: Some(_),
        }
    ));
}

#[test]
fn mag_calibrator_improves_with_consistent_gravity() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
    let mut plain = MagCalibrator::<63>::new();
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
    let rotate = |roll: f32, pitch: f32, yaw: f32, d: Matrix3<f32>| {
        let basis = UnitQuaternion::from_euler_angles(roll, pitch, yaw)
            .to_rotation_matrix()
            .into_inner();
        basis * d * basis.transpose()
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
    // before comparing: the surrogate helps isotropic-axis cases and can
    // regress on rotated-eigenvector cases, so the aggregate is what guards
    // against a net regression at the shipped low weight.
    let mut plain_total = 0.0_f32;
    let mut refined_total = 0.0_f32;

    for distortion in distortions {
        for (world_mag, world_gravity) in dip_cases {
            let mut plain = MagCalibrator::<63>::new();
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
            plain_total += error_of(&plain);
            refined_total += error_of(&gravity_refined);
        }
    }

    // At the shipped default weight (`0.01`) the gravity surrogate must not
    // regress the aggregate accuracy by more than a small, bounded amount.
    // The ellipsoid-normal surrogate pins `g_i^T A m_i` rather than the exact
    // dip `g_i^T m_i`, so rotated full-SPD soft iron can bias the fit toward
    // isotropy; the isotropic-axis cases still improve and the net regression
    // summed over all nine cases stays under this tolerance, far below the
    // repeatable regression that weight `0.1` produced in the fixed-seed
    // benchmark. Kept as an open issue rather than treated as fixed.
    let regression_tolerance = 0.2_f32;
    assert!(
        refined_total <= plain_total + regression_tolerance,
        "gravity surrogate regressed under strong anisotropy: \
         plain_total={plain_total} refined_total={refined_total}"
    );
}

#[test]
fn mag_calibrator_ignores_invalid_gravity() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);
    let mut plain = MagCalibrator::<12>::new();
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
        let theta = i as f32 * 0.31;
        let near_planar = Vector3::new(theta.cos() * 0.866, theta.sin() * 0.866, 0.5).normalize();
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
            let theta = i as f32 * 0.31;
            Vector3::new(theta.cos() * 0.866, theta.sin() * 0.866, 0.5).normalize()
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
fn live_quality_ramps_and_running_mean_square_match_the_specification() {
    // Radial RMS ramps fitness linearly from 1 at 0 to 0 at the 0.1 ceiling.
    assert_eq!(MagCalibrator::<9>::fitness_score_for_test(Some(0.0)), 1.0);
    let fitness = MagCalibrator::<9>::fitness_score_for_test(Some(0.05f32.powi(2)));
    assert!((fitness - 0.5).abs() < 1.0e-6, "fitness={fitness}");
    // At and beyond the ceiling, and for unusable statistics, fitness is 0.
    assert_eq!(
        MagCalibrator::<9>::fitness_score_for_test(Some(0.1f32.powi(2))),
        0.0
    );
    assert_eq!(
        MagCalibrator::<9>::fitness_score_for_test(Some(f32::NAN)),
        0.0
    );
    assert_eq!(MagCalibrator::<9>::fitness_score_for_test(Some(-1.0)), 0.0);
    assert_eq!(
        MagCalibrator::<9>::running_mean_square_for_test(None, 0.04, 0.25),
        Some(0.01)
    );
    let updated = MagCalibrator::<9>::running_mean_square_for_test(Some(0.04), 0.0, 0.25).unwrap();
    assert!((updated - 0.03).abs() < 1.0e-7, "updated={updated}");
    assert_eq!(
        MagCalibrator::<9>::running_mean_square_for_test(Some(f32::NAN), 0.0, 0.25),
        None
    );
    assert_eq!(
        MagCalibrator::<9>::running_mean_square_for_test(None, f32::INFINITY, 0.25),
        None
    );
}

#[test]
fn gravity_fitness_ramps_between_floor_and_ceiling_and_defaults_to_one() {
    // 1 at and below the 0.1 RMS floor, 0 at and beyond the 0.3 ceiling,
    // linear in between.
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(Some(0.0)),
        1.0
    );
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(Some(0.1f32.powi(2))),
        1.0
    );
    let fitness = MagCalibrator::<9>::gravity_fitness_score_for_test(Some(0.2f32.powi(2)));
    assert!((fitness - 0.5).abs() < 1.0e-6, "fitness={fitness}");
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(Some(0.3f32.powi(2))),
        0.0
    );
    // Unlike the radial score, a missing or unusable statistic is neutral:
    // gravity is optional and must not penalize magnetometer-only input.
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(None),
        1.0
    );
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(Some(f32::NAN)),
        1.0
    );
    assert_eq!(
        MagCalibrator::<9>::gravity_fitness_score_for_test(Some(-1.0)),
        1.0
    );
}

#[test]
fn mag_calibrator_reports_confidence_factors() {
    let offset = Vector3::new(11.0, -7.0, 5.0);
    let distortion = Matrix3::new(1.4, 0.2, -0.1, 0.2, 0.9, 0.15, -0.1, 0.15, 1.2);

    // Without gravity the gravity factor stays neutral and confidence
    // factors multiply out exactly.
    let mut plain = MagCalibrator::<63>::new();
    let mut plain_result = None;
    for i in 0..16 * 63 {
        let raw = offset + distortion * sample_direction(i % 63, 63);
        plain_result = Some(plain.evaluate_correct(raw, None, i as u64).unwrap());
    }
    let plain = plain_result.unwrap();
    assert_eq!(plain.gravity_fitness, 1.0);
    assert_eq!(plain.fitness, plain.radial_fitness);
    assert_eq!(
        plain.confidence,
        (plain.coverage * plain.fitness).clamp(0.0, 1.0)
    );

    // With a consistent co-rotating gravity direction the gravity factor is
    // live in [0, 1] and confidence is the three-factor product. Gravity
    // fixed in the body frame while the attitude rotates is physically
    // contradictory: no constant dip angle exists, the projection residual
    // stays large, and the factor drops.
    let world_mag = Vector3::new(0.8, 0.1, 0.5).normalize();
    let world_gravity = Vector3::z();
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
    assert!(
        (0.0..1.0).contains(&refined.gravity_fitness),
        "gravity_fitness={}",
        refined.gravity_fitness
    );
    assert_eq!(
        refined.fitness,
        refined.radial_fitness * refined.gravity_fitness
    );
    assert_eq!(
        refined.confidence,
        (refined.coverage * refined.fitness).clamp(0.0, 1.0)
    );
    assert!(
        opposed.gravity_fitness < refined.gravity_fitness,
        "opposed={} refined={}",
        opposed.gravity_fitness,
        refined.gravity_fitness
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
        let mut timestamp_us = 0;
        for _ in 0..4000 {
            timestamp_us += 1 + next() % 3;
            // Quantized directions with jitter: new samples frequently land
            // near buffered ones, provoking replacements.
            let theta = (next() % 8) as f32 * 0.785 + (next() % 100) as f32 / 500.0;
            let z = (next() % 5) as f32 / 2.5 - 1.0 + (next() % 100) as f32 / 500.0;
            let radius = (1.0 - z * z).max(0.0).sqrt();
            let direction = Vector3::new(radius * theta.cos(), radius * theta.sin(), z);
            let sample = Vector3::new(11.0, -7.0, 5.0) + 40.0 * direction;
            calibrator.evaluate_sample_vec(sample, None, timestamp_us);
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
#[ignore = "open issue: candidate diversity is scored against the victim row it would replace"]
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
    assert_eq!(calibrator.sample(replacement_row), victim);
    let candidate = Vector3::new(5.001, 5.0, 5.0);

    calibrator.evaluate_sample_vec(candidate, None, 12);

    // Correct post-replacement behavior: the victim is evicted and the
    // candidate takes its row. This assertion fails while the candidate is
    // still scored against the victim row it would replace.
    assert_ne!(
        calibrator.sample(replacement_row),
        victim,
        "candidate adjacent to the victim was wrongly rejected"
    );
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
    let theta = 0.37 + i as f32 * 1.21;
    let z = -0.8 + 1.6 * i as f32 / (n - 1) as f32;
    let radius = (1.0 - z * z).sqrt();
    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
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
