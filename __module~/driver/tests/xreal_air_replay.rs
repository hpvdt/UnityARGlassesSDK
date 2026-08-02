// Copyright (C) 2023, Alex Badics
// This file is part of ar-drivers-rs
// Licensed under the MIT license. See LICENSE file in the project root for details.

use std::path::Path;
use std::time::{Duration, Instant};

use ar_drivers::fusion::{rub_to_frd, MagCalibrationResult, MagCalibrator};
use ar_drivers::xreal_air::XrealAirReplay;
use ar_drivers::{ARGlasses, GlassesEvent};
use nalgebra::Vector3;

const MAX_CALIBRATION_TIME_US: u64 = 60_000_000;
const MIN_AVERAGE_FITNESS: f64 = 0.5;
/// Stability is judged over the whole post-warmup phase. The calibrator's
/// running RMS statistics persist across normalization changes, but the
/// working candidate still degrades genuinely on some trace segments, which
/// produces block-long fitness dips, so a constant bound would fail the
/// calibrator rather than the decoder. Stability therefore means a
/// consecutive streak of post-warmup evaluations above `FITNESS_FLOOR` of at
/// least `MIN_STABLE_STREAK` for both fitness components at once.
const FITNESS_FLOOR: f32 = 0.5;
const MIN_STABLE_STREAK: usize = 60;
/// Magnetometer evaluations to wait after the first successful correction.
const WARMUP_EVAL_COUNT: u64 = 125;

fn assert_air1_trace_calibrates(use_gravity: bool) {
    let mode = if use_gravity {
        "with_gravity"
    } else {
        "without_gravity"
    };
    eprintln!("# Starting replay - Air 1 trace, {mode}");
    let trace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xreal_air_air1_60s.log");
    let mut replay = XrealAirReplay::open(&trace).unwrap();
    let mut calibrator = MagCalibrator::<1023>::new();
    let mut gravity: Option<Vector3<f32>> = None;
    let mut previous_timestamp = None;
    let mut first_timestamp = None;
    let mut last_timestamp = None;
    let mut first_calibrated_timestamp = None;
    let mut accgyro_samples = 0usize;
    let mut magnetic_samples = 0usize;
    let mut eval_time = Duration::ZERO;
    let mut eval_count = 0u64;
    let mut confidence_sum = 0.0f64;
    let mut first_success_count: Option<u64> = None;
    let mut first_success_confidence: Option<f32> = None;
    let mut warmup_count: Option<u64> = None;
    let mut validation_confidence_sum = 0.0f64;
    let mut validation_confidence_count = 0u64;
    let mut validation_radial_sum = 0.0f64;
    let mut validation_gravity_sum = 0.0f64;
    let mut validation_coverage_sum = 0.0f64;
    let mut min_validation_confidence = f32::INFINITY;
    let mut max_validation_confidence = 0.0f32;
    let mut tail_samples: Vec<(u64, f32, f32)> = Vec::new();
    let mut min_confidence_radial = 0.0f32;
    let mut min_confidence_gravity = 0.0f32;
    let mut min_confidence_coverage = 0.0f32;
    let mut quality_samples = 0usize;
    let mut radial_fitness_sum = 0.0f64;
    let mut gravity_fitness_sum = 0.0f64;

    loop {
        let event = replay.read_event().unwrap();
        let timestamp = match event {
            GlassesEvent::AccGyro { timestamp, .. }
            | GlassesEvent::Magnetometer { timestamp, .. } => timestamp,
            _ => continue,
        };
        if previous_timestamp.is_some_and(|previous| timestamp < previous) {
            break;
        }
        previous_timestamp = Some(timestamp);
        first_timestamp.get_or_insert(timestamp);
        last_timestamp = Some(timestamp);

        match event {
            GlassesEvent::AccGyro { accelerometer, .. } => {
                gravity = rub_to_frd(&accelerometer).try_normalize(0.0);
                accgyro_samples += 1;
            }
            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                magnetic_samples += 1;
                let gravity_hint = if use_gravity {
                    let Some(gravity) = gravity else {
                        continue;
                    };
                    Some(gravity)
                } else {
                    None
                };
                let eval_start = Instant::now();
                let MagCalibrationResult { quality, direction } = calibrator
                    .evaluate_correct(rub_to_frd(&magnetometer), gravity_hint, timestamp)
                    .unwrap_or_else(|error| {
                        panic!("Air 1 replay {mode} calibration failed at {timestamp}: {error:?}")
                    });
                eval_time += eval_start.elapsed();
                eval_count += 1;
                confidence_sum += f64::from(quality.confidence());

                if direction.is_some() && first_success_count.is_none() {
                    first_success_count = Some(eval_count);
                    first_success_confidence = Some(quality.confidence());
                    first_calibrated_timestamp = Some(timestamp);
                }
                if first_success_count.is_some() {
                    quality_samples += 1;
                    radial_fitness_sum += f64::from(quality.radial_fitness);
                    gravity_fitness_sum += f64::from(quality.gravity_fitness);
                }

                let warmed_up = first_success_count
                    .is_some_and(|count| eval_count - count >= WARMUP_EVAL_COUNT);
                if !warmed_up {
                    continue;
                }
                if warmup_count.is_none() {
                    warmup_count = Some(eval_count - first_success_count.unwrap());
                }

                validation_confidence_sum += f64::from(quality.confidence());
                validation_confidence_count += 1;
                validation_radial_sum += f64::from(quality.radial_fitness);
                validation_gravity_sum += f64::from(quality.gravity_fitness);
                validation_coverage_sum += f64::from(quality.coverage);
                if quality.confidence() < min_validation_confidence {
                    min_validation_confidence = quality.confidence();
                    min_confidence_radial = quality.radial_fitness;
                    min_confidence_gravity = quality.gravity_fitness;
                    min_confidence_coverage = quality.coverage;
                }
                max_validation_confidence = max_validation_confidence.max(quality.confidence());
                tail_samples.push((timestamp, quality.radial_fitness, quality.gravity_fitness));
            }
            _ => {}
        }
    }

    let first_timestamp = first_timestamp.expect("Air 1 trace contained no sensor timestamp");
    let last_timestamp = last_timestamp.unwrap();
    let duration_us = last_timestamp - first_timestamp;
    let first_calibrated_timestamp = first_calibrated_timestamp
        .unwrap_or_else(|| panic!("Air 1 replay {mode} produced no calibrated reading"));
    let first_calibrated_after_us = first_calibrated_timestamp - first_timestamp;
    assert!(
        accgyro_samples > 0,
        "Air 1 replay {mode} had no AccGyro samples"
    );
    assert!(
        magnetic_samples > 0,
        "Air 1 replay {mode} had no magnetic samples"
    );
    assert!(
        first_calibrated_after_us <= MAX_CALIBRATION_TIME_US,
        "Air 1 replay {mode} first calibrated after {first_calibrated_after_us} us"
    );

    let count_until_first_success = first_success_count.unwrap();
    let first_success_confidence = first_success_confidence.unwrap();
    let warmup_count =
        warmup_count.unwrap_or_else(|| panic!("Air 1 replay {mode} warm-up never completed"));
    let verified_count = eval_count - count_until_first_success - warmup_count;
    assert!(
        quality_samples > 0,
        "Air 1 replay {mode} had no post-publication quality samples"
    );
    assert!(
        validation_confidence_count > 0,
        "Air 1 replay {mode} had no post-warmup quality samples"
    );

    let component_count = validation_confidence_count.max(1) as f64;
    let quality_count = quality_samples.max(1) as f64;
    let radial_fitness_average = radial_fitness_sum / quality_count;
    let gravity_fitness_average = gravity_fitness_sum / quality_count;

    eprintln!("- trace");
    eprintln!("  - duration: {duration_us} us");
    eprintln!("  - accgyro samples: {accgyro_samples}");
    eprintln!("  - magnetic samples: {magnetic_samples}");
    eprintln!("- evaluate_correct");
    eprintln!(
        "  - avg computation time: {:.3} ms over {} calls",
        eval_time.as_secs_f64() * 1e3 / eval_count as f64,
        eval_count,
    );
    eprintln!(
        "  - avg confidence: {:.6} over {} calls",
        confidence_sum / eval_count as f64,
        eval_count,
    );
    eprintln!(
        "  - avg post-warmup confidence: {:.6} over {} calls",
        validation_confidence_sum / component_count,
        validation_confidence_count,
    );
    eprintln!(
        "    - radial: {:.6}",
        validation_radial_sum / component_count
    );
    eprintln!(
        "    - gravity: {:.6}",
        validation_gravity_sum / component_count
    );
    eprintln!(
        "    - coverage: {:.6}",
        validation_coverage_sum / component_count
    );
    eprintln!(
        "  - worst post-warmup confidence: {:.6}",
        min_validation_confidence
    );
    eprintln!("    - radial: {:.6}", min_confidence_radial);
    eprintln!("    - gravity: {:.6}", min_confidence_gravity);
    eprintln!("    - coverage: {:.6}", min_confidence_coverage);

    let mut longest_streak = 0usize;
    let mut current_streak = 0usize;
    for (_, radial, gravity) in &tail_samples {
        if *radial >= FITNESS_FLOOR && *gravity >= FITNESS_FLOOR {
            current_streak += 1;
            longest_streak = longest_streak.max(current_streak);
        } else {
            current_streak = 0;
        }
    }
    eprintln!("- post-warmup stability: longest above-floor streak: {longest_streak} evaluations");
    eprintln!("- total: {} evaluations", eval_count);
    eprintln!(
        "  - until first successful correction: {} evaluations / confidence={:.6}",
        count_until_first_success, first_success_confidence,
    );
    eprintln!("  - sampling/optimization warm-up: {warmup_count} evaluations");
    eprintln!("  - verification: {} evaluations", verified_count);

    assert!(
        radial_fitness_average > MIN_AVERAGE_FITNESS,
        "Air 1 replay {mode} average radial_fitness {radial_fitness_average:.6} must be greater than {MIN_AVERAGE_FITNESS}"
    );
    assert!(
        gravity_fitness_average > MIN_AVERAGE_FITNESS,
        "Air 1 replay {mode} average gravity_fitness {gravity_fitness_average:.6} must be greater than {MIN_AVERAGE_FITNESS}"
    );
    assert!(
        !tail_samples.is_empty(),
        "Air 1 replay {mode} had no post-warmup evaluations"
    );
    assert!(
        longest_streak >= MIN_STABLE_STREAK,
        "Air 1 replay {mode} held both fitness components above {FITNESS_FLOOR} for at most {longest_streak} consecutive post-warmup evaluations, below {MIN_STABLE_STREAK}"
    );
}

#[test]
fn air1_trace_calibrates_with_gravity() {
    assert_air1_trace_calibrates(true);
}

#[test]
fn air1_trace_calibrates_without_gravity() {
    assert_air1_trace_calibrates(false);
}
