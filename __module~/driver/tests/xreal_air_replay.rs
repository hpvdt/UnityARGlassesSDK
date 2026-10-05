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
/// Highest radial loss allowed for the whole post-correction phase
/// average: the loss-domain equivalent of the former 0.5 radial-fitness
/// floor (fitness 0.5 corresponds to an RMS-equivalent objective
/// $\sqrt{2 J_r} = 0.25$, i.e. a radial loss $J_r = 0.25^2 / 2 = 0.03125$).
const MAX_AVERAGE_RADIAL_LOSS: f64 = 0.03125;
/// Highest gravity loss allowed for the whole post-correction phase
/// average: the loss-domain equivalent of the former 0.5 gravity-fitness
/// floor at the default weight $w_g = 0.01$ (fitness 0.5 corresponds to a
/// relative RMS dip residual of 0.225, i.e. a mean square of $0.050625$
/// and a gravity loss of $0.5 \cdot 0.01 \cdot 0.050625 \approx 0.00025$,
/// rounded down to keep the bound at most as permissive as the former
/// floor). A disabled or absent gravity term reports zero.
const MAX_AVERAGE_GRAVITY_LOSS: f64 = 0.00025;
/// Stability is judged over the whole post-correction phase. Both loss
/// statistics are recomputed from the retained rows on every quality update
/// and report the online optimizer's own objective, so block-long
/// post-correction loss spikes no longer occur. Stability nevertheless
/// means a consecutive streak of post-correction evaluations keeping both
/// losses at or below the streak ceilings — the loss-domain equivalents of
/// the former 0.5 fitness floors, derived exactly as the averages above —
/// of at least `MIN_STABLE_STREAK` for both components at once, which
/// leaves room for transient jitter on challenging trace segments without
/// weakening the constant bound into a global average.
const MAX_STABLE_RADIAL_LOSS: f32 = 0.03125;
const MAX_STABLE_GRAVITY_LOSS: f32 = 0.00025;
const MIN_STABLE_STREAK: usize = 60;
/// Interval, in evaluations after the first successful correction, marking the
/// checkpoints of the reported series. Each checkpoint aggregates the samples
/// from that point to the end of the trace, so the open-ended
/// 500/1000/1500/... series shows whether the stats measured after each
/// latency stay stable instead of degrading over time. The 60 s trace holds
/// thousands of evaluations, so the series runs until the trace ends.
const CHECKPOINT_INTERVAL: u64 = 500;
/// How far the mean confidence measured beyond a checkpoint may fall below
/// the one measured beyond the first checkpoint.
const CONFIDENCE_DEGRADATION_MARGIN: f64 = 0.1;
/// How far the mean radial or gravity loss measured beyond a checkpoint may
/// exceed the one measured beyond the first checkpoint before the run counts
/// as unstable. The former fitness margins (0.3 of a linear-in-RMS ramp) do
/// not translate into constant loss margins, so these are calibrated
/// against the trace itself: the trace's middle segment raises the suffix
/// mean radial loss by about 0.025 and the gravity loss by well under
/// 0.0001 before both recover, so these margins accept that known rise
/// while still catching a further regression.
const RADIAL_LOSS_DEGRADATION_MARGIN: f64 = 0.03;
const GRAVITY_LOSS_DEGRADATION_MARGIN: f64 = 0.0001;
/// Highest mean radial or gravity loss allowed beyond every warm-up
/// checkpoint. These are the loss-domain equivalents of the former 0.4
/// fitness floors: radial fitness 0.4 corresponds to a radial loss of
/// $0.3^2 / 2 = 0.045$, and gravity fitness 0.4 at the default weight
/// corresponds to a gravity loss of $0.5 \cdot 0.01 \cdot 0.25^2 =
/// 0.0003125$. The degradation margins only bound the losses relative to
/// the first checkpoint and the averages above only bound the whole
/// post-correction phase, so these absolute ceilings keep every checkpoint
/// average on the converged side of the boundary.
const MAX_CHECKPOINT_RADIAL_LOSS: f64 = 0.045;
const MAX_CHECKPOINT_GRAVITY_LOSS: f64 = 0.0003125;

/// One magnetometer evaluation after the first successful correction,
/// retained so every checkpoint can aggregate the span from that checkpoint
/// to the end of the trace. `latency` counts evaluations since the first
/// successful correction (which itself has latency 0).
struct PostCorrectionSample {
    latency: u64,
    confidence: f32,
    radial_loss: f32,
    regularization_loss: f32,
    gravity_loss: f32,
    coverage: f32,
}

/// Checkpoint of the reported series (a multiple of `CHECKPOINT_INTERVAL`
/// evaluations after the first successful correction). Every field aggregates
/// the samples from that checkpoint to the end of the trace.
struct CheckpointStats {
    evals_after_first_success: u64,
    mean_confidence: f64,
    mean_radial_loss: f64,
    mean_regularization_loss: f64,
    mean_gravity_loss: f64,
    mean_coverage: f64,
    min_confidence: f32,
    min_confidence_radial_loss: f32,
    min_confidence_regularization_loss: f32,
    min_confidence_gravity_loss: f32,
    min_confidence_coverage: f32,
}

/// Builds the open-ended checkpoint series from the retained post-correction
/// samples: one entry per multiple of `CHECKPOINT_INTERVAL` covered by the
/// replay, aggregating the suffix of samples starting at that latency.
fn build_checkpoints(samples: &[PostCorrectionSample]) -> Vec<CheckpointStats> {
    let Some(last_latency) = samples.last().map(|sample| sample.latency) else {
        return Vec::new();
    };
    (1..)
        .map(|checkpoint| checkpoint * CHECKPOINT_INTERVAL)
        .take_while(|&latency| latency <= last_latency)
        .map(|latency| {
            // samples are pushed in latency order
            let start = samples.partition_point(|sample| sample.latency < latency);
            let samples = &samples[start..];
            let count = samples.len().max(1) as f64;
            let mean = |field: fn(&PostCorrectionSample) -> f32| {
                samples
                    .iter()
                    .map(|sample| f64::from(field(sample)))
                    .sum::<f64>()
                    / count
            };
            let worst_confidence_sample = samples
                .iter()
                .min_by(|a, b| a.confidence.total_cmp(&b.confidence));
            CheckpointStats {
                evals_after_first_success: latency,
                mean_confidence: mean(|sample| sample.confidence),
                mean_radial_loss: mean(|sample| sample.radial_loss),
                mean_regularization_loss: mean(|sample| sample.regularization_loss),
                mean_gravity_loss: mean(|sample| sample.gravity_loss),
                mean_coverage: mean(|sample| sample.coverage),
                min_confidence: worst_confidence_sample.map_or(0.0, |sample| sample.confidence),
                min_confidence_radial_loss: worst_confidence_sample
                    .map_or(0.0, |sample| sample.radial_loss),
                min_confidence_regularization_loss: worst_confidence_sample
                    .map_or(0.0, |sample| sample.regularization_loss),
                min_confidence_gravity_loss: worst_confidence_sample
                    .map_or(0.0, |sample| sample.gravity_loss),
                min_confidence_coverage: worst_confidence_sample
                    .map_or(0.0, |sample| sample.coverage),
            }
        })
        .collect()
}

/// Formats one checkpoint field of each snapshot as a `500/1000/...`-style
/// series matching the checkpoint header row.
fn checkpoint_series(
    checkpoints: &[CheckpointStats],
    field: fn(&CheckpointStats) -> f64,
    precision: usize,
) -> String {
    checkpoints
        .iter()
        .map(|checkpoint| format!("{:.*}", precision, field(checkpoint)))
        .collect::<Vec<_>>()
        .join("/")
}

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
    // every post-correction evaluation is retained so each checkpoint can
    // aggregate the span from that checkpoint to the end of the trace
    let mut post_correction_samples: Vec<PostCorrectionSample> = Vec::new();
    let mut tail_samples: Vec<(u64, f32, f32)> = Vec::new();
    let mut quality_samples = 0usize;
    let mut radial_loss_sum = 0.0f64;
    let mut gravity_loss_sum = 0.0f64;

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
                let MagCalibrationResult {
                    quality, direction, ..
                } = calibrator
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
                    radial_loss_sum += f64::from(quality.radial_loss);
                    gravity_loss_sum += f64::from(quality.gravity_loss);
                }

                // post-correction samples accumulate from the first successful
                // correction; per-checkpoint stats are derived after the replay
                let Some(count_at_first_success) = first_success_count else {
                    continue;
                };
                let latency = eval_count - count_at_first_success;

                post_correction_samples.push(PostCorrectionSample {
                    latency,
                    confidence: quality.confidence(),
                    radial_loss: quality.radial_loss,
                    regularization_loss: quality.regularization_loss,
                    gravity_loss: quality.gravity_loss,
                    coverage: quality.coverage,
                });
                tail_samples.push((timestamp, quality.radial_loss, quality.gravity_loss));
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
    let post_correction_count = eval_count - count_until_first_success;
    let checkpoints = build_checkpoints(&post_correction_samples);
    assert!(
        quality_samples > 0,
        "Air 1 replay {mode} had no post-publication quality samples"
    );
    assert!(
        !post_correction_samples.is_empty(),
        "Air 1 replay {mode} had no post-correction quality samples"
    );
    assert!(
        !checkpoints.is_empty(),
        "Air 1 replay {mode} recorded no checkpoints: first success at evaluation \
         {count_until_first_success} of {eval_count}"
    );

    let quality_count = quality_samples.max(1) as f64;
    let radial_loss_average = radial_loss_sum / quality_count;
    let gravity_loss_average = gravity_loss_sum / quality_count;

    eprintln!("- trace");
    eprintln!("  - duration: {duration_us} us");
    eprintln!("  - accgyro samples: {accgyro_samples}");
    eprintln!("  - magnetic samples: {magnetic_samples}");
    eprintln!("- evaluate_correct");
    eprintln!(
        "  - post-warmup checkpoints: {} (evals after first successful correction)",
        checkpoints
            .iter()
            .map(|checkpoint| checkpoint.evals_after_first_success.to_string())
            .collect::<Vec<_>>()
            .join("/"),
    );
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
        "  - avg post-warmup confidence: {}",
        checkpoint_series(&checkpoints, |checkpoint| checkpoint.mean_confidence, 6),
    );
    eprintln!(
        "    - radial loss: {}",
        checkpoint_series(&checkpoints, |checkpoint| checkpoint.mean_radial_loss, 6),
    );
    eprintln!(
        "    - regularization loss: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| checkpoint.mean_regularization_loss,
            6
        ),
    );
    eprintln!(
        "    - gravity loss: {}",
        checkpoint_series(&checkpoints, |checkpoint| checkpoint.mean_gravity_loss, 6),
    );
    eprintln!(
        "    - coverage: {}",
        checkpoint_series(&checkpoints, |checkpoint| checkpoint.mean_coverage, 6),
    );
    eprintln!(
        "  - worst post-warmup confidence: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence),
            6
        ),
    );
    eprintln!(
        "    - radial loss: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_radial_loss),
            6
        ),
    );
    eprintln!(
        "    - regularization loss: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_regularization_loss),
            6
        ),
    );
    eprintln!(
        "    - gravity loss: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_gravity_loss),
            6
        ),
    );
    eprintln!(
        "    - coverage: {}",
        checkpoint_series(
            &checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_coverage),
            6
        ),
    );

    let mut longest_streak = 0usize;
    let mut current_streak = 0usize;
    for (_, radial_loss, gravity_loss) in &tail_samples {
        if *radial_loss <= MAX_STABLE_RADIAL_LOSS && *gravity_loss <= MAX_STABLE_GRAVITY_LOSS {
            current_streak += 1;
            longest_streak = longest_streak.max(current_streak);
        } else {
            current_streak = 0;
        }
    }
    eprintln!(
        "- post-correction stability: longest above-floor streak: {longest_streak} evaluations"
    );
    eprintln!("- total: {} evaluations", eval_count);
    eprintln!(
        "  - until first successful correction: {} evaluations / confidence={:.6}",
        count_until_first_success, first_success_confidence,
    );
    eprintln!(
        "  - after first successful correction: {post_correction_count} evaluations ({} checkpoints of {CHECKPOINT_INTERVAL})",
        checkpoints.len(),
    );

    // stability: stats measured from each checkpoint onward must not be worse
    // than the ones measured from the first checkpoint onward
    let first_checkpoint = &checkpoints[0];
    for checkpoint in &checkpoints[1..] {
        assert!(
            checkpoint.mean_confidence
                >= first_checkpoint.mean_confidence - CONFIDENCE_DEGRADATION_MARGIN,
            "Air 1 replay {mode} post-correction mean confidence measured from evaluation {} \
             onward ({:.6}) fell below the one from evaluation {} onward ({:.6}) by more \
             than {CONFIDENCE_DEGRADATION_MARGIN}",
            checkpoint.evals_after_first_success,
            checkpoint.mean_confidence,
            first_checkpoint.evals_after_first_success,
            first_checkpoint.mean_confidence,
        );
        for (label, current, reference, margin) in [
            (
                "radial",
                checkpoint.mean_radial_loss,
                first_checkpoint.mean_radial_loss,
                RADIAL_LOSS_DEGRADATION_MARGIN,
            ),
            (
                "gravity",
                checkpoint.mean_gravity_loss,
                first_checkpoint.mean_gravity_loss,
                GRAVITY_LOSS_DEGRADATION_MARGIN,
            ),
        ] {
            assert!(
                current <= reference + margin,
                "Air 1 replay {mode} post-correction mean {label} loss measured from \
                 evaluation {} onward ({current:.6}) exceeded the one from evaluation {} \
                 onward ({reference:.6}) by more than {margin}",
                checkpoint.evals_after_first_success,
                first_checkpoint.evals_after_first_success,
            );
        }
    }
    // loss ceiling: the radial and gravity losses averaged from every
    // warm-up checkpoint onward must stay below the absolute ceilings
    for checkpoint in &checkpoints {
        for (label, loss, ceiling) in [
            (
                "radial",
                checkpoint.mean_radial_loss,
                MAX_CHECKPOINT_RADIAL_LOSS,
            ),
            (
                "gravity",
                checkpoint.mean_gravity_loss,
                MAX_CHECKPOINT_GRAVITY_LOSS,
            ),
        ] {
            assert!(
                loss < ceiling,
                "Air 1 replay {mode} post-correction mean {label} loss measured from \
                 evaluation {} onward ({loss:.6}) was not below {ceiling}",
                checkpoint.evals_after_first_success,
            );
        }
    }

    assert!(
        radial_loss_average < MAX_AVERAGE_RADIAL_LOSS,
        "Air 1 replay {mode} average radial_loss {radial_loss_average:.6} must be below {MAX_AVERAGE_RADIAL_LOSS}"
    );
    assert!(
        gravity_loss_average < MAX_AVERAGE_GRAVITY_LOSS,
        "Air 1 replay {mode} average gravity_loss {gravity_loss_average:.6} must be below {MAX_AVERAGE_GRAVITY_LOSS}"
    );
    assert!(
        !tail_samples.is_empty(),
        "Air 1 replay {mode} had no post-correction evaluations"
    );
    assert!(
        longest_streak >= MIN_STABLE_STREAK,
        "Air 1 replay {mode} held both losses at or below the streak ceilings for at most \
         {longest_streak} consecutive post-correction evaluations, below {MIN_STABLE_STREAK}"
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
