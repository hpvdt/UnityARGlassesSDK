use std::time::{Duration, Instant};

use ar_drivers::fusion::{rub_to_frd, FusionState};
use ar_drivers::sim::sim_motion::Config;
use ar_drivers::{ARGlasses, GlassesEvent, SimMotion};
use nalgebra::Vector3;
use serial_test::serial;

const CONFIDENCE_THRESHOLD: f32 = 0.4;
/// Unpaced virtual period: SimMotion skips the wall-clock pacing sleep once the
/// event period exceeds the built-in 20 ms pacing bound, so the benchmark runs
/// as fast as the hardware allows. Zero would freeze the attitude simulation
/// (dt = `event_period_us` seconds), so it must stay positive.
const EVENT_PERIOD_US: u64 = 20_001;
/// Total magnetometer evaluations per run. The loop always runs for exactly
/// this many evaluations (doubling as the hang guard), so the post-correction
/// phase holds several checkpoints even when the first successful correction
/// arrives late (slowest observed: ~1200 evaluations).
const MAX_EVAL_COUNT: u64 = 3_000;
/// Interval, in evaluations after the first successful correction, marking the
/// checkpoints of the reported series. Each checkpoint aggregates the samples
/// from that point to the end of the run, so the open-ended 500/1000/1500/...
/// series shows whether the stats measured after each latency stay stable
/// instead of degrading over time.
const CHECKPOINT_INTERVAL: u64 = 500;

/// Worst single correction error allowed beyond the first checkpoint.
const WORST_VALIDATION_ERROR_CRITERION: f32 = 25.0;

/// Highest average correction error allowed beyond the first checkpoint.
const AVG_VALIDATION_ERROR_AFTER_CRITERION: f64 = 10.0;

/// How far the mean error measured beyond a checkpoint may exceed the one
/// measured beyond the first checkpoint before the run counts as unstable.
const ERROR_DEGRADATION_MARGIN_DEGREES: f64 = 5.0;

/// How far the mean confidence measured beyond a checkpoint may fall below
/// the one measured beyond the first checkpoint.
const CONFIDENCE_DEGRADATION_MARGIN: f64 = 0.1;

/// Whether the calibrator is fed a co-timestamped simulated accelerometer reading with each sample.
#[derive(Clone, Copy)]
enum AttitudeMode {
    Always,
    Never,
}

/// One magnetometer evaluation after the first successful correction,
/// retained so every checkpoint can aggregate the span from that checkpoint
/// to the end of the run. `latency` counts evaluations since the first
/// successful correction (which itself has latency 0).
struct PostCorrectionSample {
    latency: u64,
    error_degrees: Option<f32>,
    confidence: f32,
    radial: f32,
    gravity: f32,
    coverage: f32,
}

/// Checkpoint of the reported series (a multiple of `CHECKPOINT_INTERVAL`
/// evaluations after the first successful correction). Every field aggregates
/// the samples from that checkpoint to the end of the run.
struct CheckpointStats {
    evals_after_first_success: u64,
    mean_error_degrees: f64,
    worst_error_degrees: f32,
    mean_confidence: f64,
    mean_radial: f64,
    mean_gravity: f64,
    mean_coverage: f64,
    min_confidence: f32,
    min_confidence_radial: f32,
    min_confidence_gravity: f32,
    min_confidence_coverage: f32,
}

/// Builds the open-ended checkpoint series from the retained post-correction
/// samples: one entry per multiple of `CHECKPOINT_INTERVAL` covered by the
/// run, aggregating the suffix of samples starting at that latency.
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
            let (error_sum, error_count, worst_error) = samples
                .iter()
                .filter_map(|sample| sample.error_degrees)
                .fold((0.0f64, 0u64, 0.0f32), |(sum, count, worst), error| {
                    (sum + f64::from(error), count + 1, worst.max(error))
                });
            let worst_confidence_sample = samples
                .iter()
                .min_by(|a, b| a.confidence.total_cmp(&b.confidence));
            CheckpointStats {
                evals_after_first_success: latency,
                mean_error_degrees: error_sum / error_count.max(1) as f64,
                worst_error_degrees: worst_error,
                mean_confidence: mean(|sample| sample.confidence),
                mean_radial: mean(|sample| sample.radial),
                mean_gravity: mean(|sample| sample.gravity),
                mean_coverage: mean(|sample| sample.coverage),
                min_confidence: worst_confidence_sample.map_or(0.0, |sample| sample.confidence),
                min_confidence_radial: worst_confidence_sample.map_or(0.0, |sample| sample.radial),
                min_confidence_gravity: worst_confidence_sample
                    .map_or(0.0, |sample| sample.gravity),
                min_confidence_coverage: worst_confidence_sample
                    .map_or(0.0, |sample| sample.coverage),
            }
        })
        .collect()
}

#[derive(Default)]
struct RunStats {
    eval_time: Duration,
    eval_count: u64,
    confidence_sum: f64,
    confidence_count: u64,
    error_sum_degrees: f64,
    error_count: u64,
    worst_error: f32,
    first_success_confidence: f32,
    sum_validation_error_after_warmup: f64,
    worst_validation_error_after_warmup: f32,
    validation_error_count: u64,
    validation_confidence_sum: f64,
    validation_confidence_count: u64,
    count_until_first_success: u64,
    checkpoints: Vec<CheckpointStats>,
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

fn run_calibration(config: Config, attitude_mode: AttitudeMode) -> RunStats {
    let seed = config.seed;
    let mode_label = match attitude_mode {
        AttitudeMode::Always => "with accelerometer gravity",
        AttitudeMode::Never => "without gravity",
    };
    println!("# Starting benchmark - PRNG seed: {seed}, {mode_label}");

    let dip = config
        .magnetic_dip_rad
        .clamp(-30.0f32.to_radians(), 30.0f32.to_radians());
    let magnetic_world_rub =
        Vector3::new(0.0, dip.sin(), -dip.cos()) * config.magnetic_field_strength;
    let mut sim_motion = SimMotion::with_config(config);
    let mut fusion = FusionState::new(Box::new(SimMotion::new()));
    let mut first_success_count = None;
    // every post-correction evaluation is retained so each checkpoint can
    // aggregate the span from that checkpoint to the end of the run
    let mut post_correction_samples: Vec<PostCorrectionSample> = Vec::new();

    let mut stats = RunStats::default();
    // locals below back up assert messages and first-success tracking only
    let mut max_confidence = 0.0f32;
    let mut quality_streak = 0usize;
    let mut max_quality_streak = 0usize;
    let mut last_timestamp = 0u64;
    let mut confidence_at_worst_validation_error = 0.0f32;
    let mut timestamp_at_worst_validation_error = 0u64;
    let mut count_until_first_success = None;
    let mut first_success_confidence = None;
    loop {
        if stats.eval_count >= MAX_EVAL_COUNT {
            break;
        }
        let ground_truth = sim_motion.snapshot();
        let accelerometer_frd = (!ground_truth.next_event_is_acc_gyro)
            .then(|| rub_to_frd(&sim_motion.accelerometer_reading()));
        let event = sim_motion.read_event().unwrap();
        let (magnetometer, timestamp) = match event {
            GlassesEvent::AccGyro { .. } => continue,
            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => (magnetometer, timestamp),
            _ => continue,
        };
        last_timestamp = timestamp;

        assert_eq!(timestamp, ground_truth.timestamp_us);
        let ideal_body_rub = ground_truth.attitude.inverse() * magnetic_world_rub;
        let ideal_body_frd = rub_to_frd(&ideal_body_rub).normalize();
        let raw_frd = rub_to_frd(&magnetometer);

        let gravity_direction = match attitude_mode {
            AttitudeMode::Always => Some(
                accelerometer_frd
                    .expect("sim_motion magnetometer event did not have an accelerometer sample"),
            ),
            AttitudeMode::Never => None,
        };
        let eval_start = Instant::now();
        let result = fusion
            .magCalibrator
            .evaluate_correct(raw_frd, gravity_direction, timestamp);
        stats.eval_time += eval_start.elapsed();
        stats.eval_count += 1;
        let (confidence, radial_fitness, gravity_fitness, coverage) = result.as_ref().map_or_else(
            |_| (fusion.magCalibrator.get_confidence(), 0.0, 0.0, 0.0),
            |result| {
                (
                    result.confidence(),
                    result.radial_fitness,
                    result.gravity_fitness,
                    result.coverage,
                )
            },
        );
        stats.confidence_sum += f64::from(confidence);
        stats.confidence_count += 1;
        max_confidence = max_confidence.max(confidence);
        if confidence >= CONFIDENCE_THRESHOLD {
            quality_streak += 1;
            max_quality_streak = max_quality_streak.max(quality_streak);
        } else {
            quality_streak = 0;
        }
        let corrected = result.as_ref().ok().and_then(|result| result.direction);
        let angle_degrees =
            corrected.map(|direction| direction.angle(&ideal_body_frd).to_degrees());
        if let Some(angle_degrees) = angle_degrees {
            stats.error_sum_degrees += f64::from(angle_degrees);
            stats.error_count += 1;
            stats.worst_error = stats.worst_error.max(angle_degrees);
        }

        if corrected.is_some() && first_success_count.is_none() {
            first_success_count = Some(stats.eval_count);
            count_until_first_success = Some(stats.eval_count);
            first_success_confidence = Some(confidence);
        }
        let Some(count_at_first_success) = first_success_count else {
            continue;
        };
        let latency = stats.eval_count - count_at_first_success;

        // failed evaluations carry no quality sample (the tuple above falls
        // back to zeroed components for them); exclude them. They can only
        // occur before the first checkpoint — the strict phase beyond it
        // panics on them.
        if result.is_ok() {
            post_correction_samples.push(PostCorrectionSample {
                latency,
                error_degrees: angle_degrees,
                confidence,
                radial: radial_fitness,
                gravity: gravity_fitness,
                coverage,
            });
        }

        // strict validation (worst/average error criteria and panics on
        // failed/pending corrections) applies only beyond the first checkpoint
        if latency <= CHECKPOINT_INTERVAL {
            continue;
        }
        if let Err(error) = result {
            panic!(
                "magnetometer calibration failed for seed={seed}, mode={mode_label}, \
                 timestamp={timestamp}, confidence={confidence}: {error:?}"
            )
        }
        let angle_degrees = angle_degrees.unwrap_or_else(|| {
            panic!(
                "magnetometer calibration returned pending for seed={seed}, mode={mode_label}, \
                 timestamp={timestamp}, confidence={confidence}"
            )
        });
        stats.sum_validation_error_after_warmup += f64::from(angle_degrees);
        stats.validation_error_count += 1;
        stats.validation_confidence_sum += f64::from(confidence);
        stats.validation_confidence_count += 1;
        if angle_degrees > stats.worst_validation_error_after_warmup {
            stats.worst_validation_error_after_warmup = angle_degrees;
            confidence_at_worst_validation_error = confidence;
            timestamp_at_worst_validation_error = timestamp;
        }
    }

    let count_until_first_success = count_until_first_success.unwrap_or_else(|| {
        panic!(
            "magnetometer calibration never succeeded within {MAX_EVAL_COUNT} evaluations: \
             seed={seed}, mode={mode_label}, timestamp={last_timestamp}, \
             current_confidence={}, max_confidence={max_confidence}, \
             quality_streak={quality_streak}, max_quality_streak={max_quality_streak}",
            fusion.magCalibrator.get_confidence(),
        )
    });
    stats.count_until_first_success = count_until_first_success;
    stats.first_success_confidence = first_success_confidence.unwrap();
    stats.checkpoints = build_checkpoints(&post_correction_samples);
    assert!(
        !stats.checkpoints.is_empty(),
        "magnetometer calibration recorded no checkpoints: seed={seed}, mode={mode_label}, \
         first success at evaluation {count_until_first_success} of {MAX_EVAL_COUNT}"
    );
    assert!(
        stats.validation_error_count > 0,
        "magnetometer calibration recorded no evaluations beyond the first checkpoint: \
         seed={seed}, mode={mode_label}, \
         first success at evaluation {count_until_first_success} of {MAX_EVAL_COUNT}"
    );
    // stability: stats measured from each checkpoint onward must not be worse
    // than the ones measured from the first checkpoint onward
    let first_checkpoint = &stats.checkpoints[0];
    for checkpoint in &stats.checkpoints[1..] {
        assert!(
            checkpoint.mean_error_degrees
                <= first_checkpoint.mean_error_degrees + ERROR_DEGRADATION_MARGIN_DEGREES,
            "post-correction mean error measured from evaluation {} onward ({:.3} deg) \
             exceeded the one from evaluation {} onward ({:.3} deg) by more than \
             {ERROR_DEGRADATION_MARGIN_DEGREES} deg: seed={seed}, mode={mode_label}",
            checkpoint.evals_after_first_success,
            checkpoint.mean_error_degrees,
            first_checkpoint.evals_after_first_success,
            first_checkpoint.mean_error_degrees,
        );
        assert!(
            checkpoint.mean_confidence
                >= first_checkpoint.mean_confidence - CONFIDENCE_DEGRADATION_MARGIN,
            "post-correction mean confidence measured from evaluation {} onward ({:.6}) \
             fell below the one from evaluation {} onward ({:.6}) by more than \
             {CONFIDENCE_DEGRADATION_MARGIN}: seed={seed}, mode={mode_label}",
            checkpoint.evals_after_first_success,
            checkpoint.mean_confidence,
            first_checkpoint.evals_after_first_success,
            first_checkpoint.mean_confidence,
        );
    }

    println!("- evaluate_correct");
    println!(
        "  - post-warmup checkpoints: {} (evals after first successful correction)",
        stats
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.evals_after_first_success.to_string())
            .collect::<Vec<_>>()
            .join("/"),
    );
    println!(
        "  - avg computation time: {:.3} ms over {} calls",
        stats.eval_time.as_secs_f64() * 1e3 / stats.eval_count as f64,
        stats.eval_count,
    );
    println!(
        "  - avg confidence: {:.6} over {} calls",
        stats.confidence_sum / stats.confidence_count as f64,
        stats.confidence_count,
    );
    println!(
        "  - avg error: {:.3} deg over {} successful calls",
        stats.error_sum_degrees / stats.error_count as f64,
        stats.error_count,
    );
    println!(
        "    - post-warmup: {} deg",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| checkpoint.mean_error_degrees,
            3
        ),
    );
    println!("  - worst error: {:.3} deg", stats.worst_error);
    println!(
        "    - post-warmup: {} deg",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| f64::from(checkpoint.worst_error_degrees),
            3
        ),
    );
    println!(
        "  - avg post-warmup confidence: {}",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| checkpoint.mean_confidence,
            6
        ),
    );
    println!(
        "    - radial: {}",
        checkpoint_series(&stats.checkpoints, |checkpoint| checkpoint.mean_radial, 6),
    );
    println!(
        "    - gravity: {}",
        checkpoint_series(&stats.checkpoints, |checkpoint| checkpoint.mean_gravity, 6),
    );
    println!(
        "    - coverage: {}",
        checkpoint_series(&stats.checkpoints, |checkpoint| checkpoint.mean_coverage, 6),
    );
    println!(
        "  - worst post-warmup confidence: {}",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence),
            6
        ),
    );
    println!(
        "    - radial: {}",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_radial),
            6
        ),
    );
    println!(
        "    - gravity: {}",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_gravity),
            6
        ),
    );
    println!(
        "    - coverage: {}",
        checkpoint_series(
            &stats.checkpoints,
            |checkpoint| f64::from(checkpoint.min_confidence_coverage),
            6
        ),
    );
    println!("- total: {} evaluations", stats.eval_count);
    println!(
        "  - until first successful correction: {} evaluations / confidence={:.6}",
        count_until_first_success,
        first_success_confidence.unwrap(),
    );
    println!(
        "  - after first successful correction: {} evaluations ({} checkpoints of {CHECKPOINT_INTERVAL})",
        stats.eval_count - count_until_first_success,
        stats.checkpoints.len(),
    );

    let avg_validation_error_after_warmup =
        stats.sum_validation_error_after_warmup / stats.validation_error_count.max(1) as f64;

    assert!(
        stats.worst_validation_error_after_warmup <= WORST_VALIDATION_ERROR_CRITERION,
        "worst corrected magnetometer error exceeded {WORST_VALIDATION_ERROR_CRITERION} degrees: seed={seed}, mode={mode_label}, \
         timestamp={timestamp_at_worst_validation_error}, \
         confidence={confidence_at_worst_validation_error}, \
         worst_angle_degrees={}",
        stats.worst_validation_error_after_warmup
    );
    assert!(
        avg_validation_error_after_warmup <= AVG_VALIDATION_ERROR_AFTER_CRITERION,
        "average corrected magnetometer error exceeded {AVG_VALIDATION_ERROR_AFTER_CRITERION} degrees: seed={seed}, mode={mode_label}, \
         avg_validation_confidence={}, \
         avg_validation_error_degrees={avg_validation_error_after_warmup}",
        stats.validation_confidence_sum / stats.validation_confidence_count.max(1) as f64,
    );

    stats
}

fn print_avg_stats(runs: &[RunStats]) {
    let n = runs.len() as f64;
    let avg_count =
        |f: fn(&RunStats) -> u64| (runs.iter().map(|r| f(r)).sum::<u64>() as f64 / n).round();
    let total_eval_time: f64 = runs.iter().map(|r| r.eval_time.as_secs_f64()).sum();
    let total_eval_count: u64 = runs.iter().map(|r| r.eval_count).sum();
    let total_confidence_sum: f64 = runs.iter().map(|r| r.confidence_sum).sum();
    let total_confidence_count: u64 = runs.iter().map(|r| r.confidence_count).sum();
    let total_error_sum: f64 = runs.iter().map(|r| r.error_sum_degrees).sum();
    let total_error_count: u64 = runs.iter().map(|r| r.error_count).sum();
    let worst_error_degrees: f32 = runs.iter().map(|r| r.worst_error).fold(0.0, f32::max);
    // position-wise checkpoint aggregation: runs missing a position (first
    // success arrived later) are excluded from that position
    let max_checkpoints = runs.iter().map(|r| r.checkpoints.len()).max().unwrap_or(0);
    let avg_checkpoint_series = |field: fn(&CheckpointStats) -> f64, precision: usize| {
        (0..max_checkpoints)
            .map(|position| {
                let (sum, count) = runs
                    .iter()
                    .filter_map(|r| r.checkpoints.get(position))
                    .fold((0.0, 0usize), |(sum, count), checkpoint| {
                        (sum + field(checkpoint), count + 1)
                    });
                format!(
                    "{:.*}",
                    precision,
                    if count > 0 { sum / count as f64 } else { 0.0 },
                )
            })
            .collect::<Vec<_>>()
            .join("/")
    };
    // per checkpoint position, the run whose snapshot holds the worst
    // confidence (mirrors position-wise worst across runs)
    let worst_checkpoint_series = |field: fn(&CheckpointStats) -> f64| {
        (0..max_checkpoints)
            .map(|position| {
                let worst = runs
                    .iter()
                    .filter_map(|r| r.checkpoints.get(position))
                    .min_by(|a, b| a.min_confidence.total_cmp(&b.min_confidence));
                format!("{:.6}", worst.map_or(0.0, field))
            })
            .collect::<Vec<_>>()
            .join("/")
    };

    println!("  ======================================================================  ");
    println!("# Average stats over {} runs", runs.len());
    println!("- evaluate_correct");
    println!(
        "  - post-warmup checkpoints: {} (evals after first successful correction)",
        (1..=max_checkpoints)
            .map(|position| (position as u64 * CHECKPOINT_INTERVAL).to_string())
            .collect::<Vec<_>>()
            .join("/"),
    );
    println!(
        "  - avg computation time: {:.3} ms over {} calls",
        total_eval_time * 1e3 / total_eval_count as f64,
        avg_count(|r| r.eval_count),
    );
    println!(
        "  - avg confidence: {:.6} over {} calls",
        total_confidence_sum / total_confidence_count as f64,
        avg_count(|r| r.confidence_count),
    );
    println!(
        "  - avg error: {:.3} deg over {} successful calls",
        total_error_sum / total_error_count as f64,
        avg_count(|r| r.error_count),
    );
    println!(
        "    - post-warmup: {} deg",
        avg_checkpoint_series(|checkpoint| checkpoint.mean_error_degrees, 3),
    );
    println!("  - worst error: {worst_error_degrees:.3} deg");
    println!(
        "    - post-warmup: {} deg",
        avg_checkpoint_series(|checkpoint| f64::from(checkpoint.worst_error_degrees), 3),
    );
    println!(
        "  - avg post-warmup confidence: {}",
        avg_checkpoint_series(|checkpoint| checkpoint.mean_confidence, 6),
    );
    println!(
        "    - radial: {}",
        avg_checkpoint_series(|checkpoint| checkpoint.mean_radial, 6),
    );
    println!(
        "    - gravity: {}",
        avg_checkpoint_series(|checkpoint| checkpoint.mean_gravity, 6),
    );
    println!(
        "    - coverage: {}",
        avg_checkpoint_series(|checkpoint| checkpoint.mean_coverage, 6),
    );
    println!(
        "  - worst post-warmup confidence: {}",
        worst_checkpoint_series(|checkpoint| f64::from(checkpoint.min_confidence)),
    );
    println!(
        "    - radial: {}",
        worst_checkpoint_series(|checkpoint| f64::from(checkpoint.min_confidence_radial)),
    );
    println!(
        "    - gravity: {}",
        worst_checkpoint_series(|checkpoint| f64::from(checkpoint.min_confidence_gravity)),
    );
    println!(
        "    - coverage: {}",
        worst_checkpoint_series(|checkpoint| f64::from(checkpoint.min_confidence_coverage)),
    );
    println!("- total: {} evaluations", avg_count(|r| r.eval_count));
    println!(
        "  - until first successful correction: {} evaluations / avg confidence={:.6}",
        avg_count(|r| r.count_until_first_success),
        runs.iter()
            .map(|r| f64::from(r.first_success_confidence))
            .sum::<f64>()
            / n,
    );
    println!(
        "  - after first successful correction: {} evaluations ({} checkpoints of {CHECKPOINT_INTERVAL})",
        avg_count(|r| r.eval_count - r.count_until_first_success),
        runs.iter().map(|r| r.checkpoints.len() as u64).sum::<u64>() as f64 / n,
    );
}

/// Runs each seed with the given attitude mode, printing average stats.
fn run_seeds(attitude_mode: AttitudeMode, seeds: impl IntoIterator<Item = u64>) {
    let runs: Vec<RunStats> = seeds
        .into_iter()
        .map(|seed| {
            let config = Config {
                seed,
                event_period_us: EVENT_PERIOD_US,
                ..Config::default()
            };
            run_calibration(config, attitude_mode)
        })
        .collect();
    print_avg_stats(&runs);
}

#[test_case::test_case(AttitudeMode::Never  ; "without_gravity")]
#[test_case::test_case(AttitudeMode::Always ; "with_gravity")]
#[serial]
fn short(attitude_mode: AttitudeMode) {
    run_seeds(attitude_mode, [rand::random()]);
}

#[test_case::test_case(AttitudeMode::Never  ; "without_gravity")]
#[test_case::test_case(AttitudeMode::Always ; "with_gravity")]
#[serial]
fn long(attitude_mode: AttitudeMode) {
    run_seeds(attitude_mode, (0..20).map(|_| rand::random()));
}

#[test_case::test_case(AttitudeMode::Never  ; "without_gravity")]
#[test_case::test_case(AttitudeMode::Always ; "with_gravity")]
#[serial]
fn regression(attitude_mode: AttitudeMode) {
    run_seeds(
        attitude_mode,
        [
            934786981548549007,
            320366629120039532,
            800448092538851856,
            14346460742415463748,
            308857554940434960,
            9627152797423610735,
            15214809500125664723,
            4333660961526349397,
            17611800246992533302,
            9399375230094018656,
            10758804304863325866,
        ],
    );
}
