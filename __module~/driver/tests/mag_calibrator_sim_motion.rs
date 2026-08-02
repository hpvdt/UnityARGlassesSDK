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
/// Hang guard bounding the whole benchmark in magnetometer evaluations.
/// Leaves headroom over the slowest observed fixed-seed run (under 1800
/// evaluations with the 55-update publication streak).
const MAX_EVAL_COUNT: u64 = 2_000;
/// Magnetometer evaluations to wait after the first successful correction.
const WARMUP_EVAL_COUNT: u64 = 125;
/// Magnetometer evaluations to validate before the run can end early.
const REQUIRED_VALIDATION_EVAL_COUNT: u64 = 125;
/// Magnetometer evaluations to validate in total, ending the run.
const VALIDATION_EVAL_COUNT: u64 = 500;

const WORST_VALIDATION_ERROR_CRITERION: f32 = 25.0;

const AVG_VALIDATION_ERROR_AFTER_CRITERION: f64 = 10.0;

/// Whether the calibrator is fed a co-timestamped simulated accelerometer reading with each sample.
#[derive(Clone, Copy)]
enum AttitudeMode {
    Always,
    Never,
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
    validation_radial_sum: f64,
    validation_gravity_sum: f64,
    validation_coverage_sum: f64,
    min_validation_confidence: f32,
    max_validation_confidence: f32,
    min_confidence_radial: f32,
    min_confidence_gravity: f32,
    min_confidence_coverage: f32,
    count_until_first_success: u64,
    warmup_count: u64,
    verified_count: u64,
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
    let mut validation_start_count = None;
    let mut completed_required_validation = false;

    let mut stats = RunStats {
        min_validation_confidence: f32::INFINITY,
        ..RunStats::default()
    };
    // locals below back up assert messages and warmup/first-success tracking only
    let mut max_confidence = 0.0f32;
    let mut quality_streak = 0usize;
    let mut max_quality_streak = 0usize;
    let mut last_timestamp = 0u64;
    let mut confidence_at_worst_validation_error = 0.0f32;
    let mut timestamp_at_worst_validation_error = 0u64;
    let mut count_until_first_success = None;
    let mut first_success_confidence = None;
    let mut warmup_count = None;
    loop {
        assert!(
            stats.eval_count <= MAX_EVAL_COUNT,
            "magnetometer calibration never succeeded within {MAX_EVAL_COUNT} evaluations: \
             seed={seed}, mode={mode_label}, timestamp={last_timestamp}, \
             current_confidence={}, max_confidence={max_confidence}, \
             quality_streak={quality_streak}, max_quality_streak={max_quality_streak}",
            fusion.magCalibrator.get_confidence(),
        );
        if validation_start_count
            .is_some_and(|start| stats.eval_count - start >= VALIDATION_EVAL_COUNT)
        {
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

        let warmed_up =
            first_success_count.is_some_and(|count| stats.eval_count - count >= WARMUP_EVAL_COUNT);
        if !warmed_up {
            if corrected.is_some() && first_success_count.is_none() {
                first_success_count = Some(stats.eval_count);
                count_until_first_success = Some(stats.eval_count);
                first_success_confidence = Some(confidence);
            }
            continue;
        }
        if warmup_count.is_none() {
            let count_at_first_success = first_success_count.unwrap();
            warmup_count = Some(stats.eval_count - count_at_first_success);
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
        stats.validation_radial_sum += f64::from(radial_fitness);
        stats.validation_gravity_sum += f64::from(gravity_fitness);
        stats.validation_coverage_sum += f64::from(coverage);
        if confidence < stats.min_validation_confidence {
            stats.min_validation_confidence = confidence;
            stats.min_confidence_radial = radial_fitness;
            stats.min_confidence_gravity = gravity_fitness;
            stats.min_confidence_coverage = coverage;
        }
        stats.max_validation_confidence = stats.max_validation_confidence.max(confidence);
        let start = *validation_start_count.get_or_insert(stats.eval_count);
        if angle_degrees > stats.worst_validation_error_after_warmup {
            stats.worst_validation_error_after_warmup = angle_degrees;
            confidence_at_worst_validation_error = confidence;
            timestamp_at_worst_validation_error = timestamp;
        }
        if stats.eval_count - start >= REQUIRED_VALIDATION_EVAL_COUNT {
            completed_required_validation = true;
        }
    }

    assert!(
        completed_required_validation,
        "corrected magnetometer did not complete the required {REQUIRED_VALIDATION_EVAL_COUNT}-evaluation \
         validation; worst_error_after_warmup={}",
        stats.worst_validation_error_after_warmup,
    );

    let count_until_first_success = count_until_first_success.expect("no successful correction");
    let warmup_count = warmup_count.expect("warm-up never completed");
    stats.verified_count = stats.eval_count - count_until_first_success - warmup_count;
    stats.count_until_first_success = count_until_first_success;
    stats.warmup_count = warmup_count;
    stats.first_success_confidence = first_success_confidence.unwrap();

    println!("- evaluate_correct");
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
        "    - post-warmup: {:.3} deg over {} calls",
        stats.sum_validation_error_after_warmup / stats.validation_error_count.max(1) as f64,
        stats.validation_error_count,
    );
    println!("  - worst error: {:.3} deg", stats.worst_error);
    println!(
        "    - post-warmup: {:.3} deg",
        stats.worst_validation_error_after_warmup
    );
    println!(
        "  - avg post-warmup confidence: {:.6} over {} calls",
        stats.validation_confidence_sum / stats.validation_confidence_count.max(1) as f64,
        stats.validation_confidence_count,
    );
    let component_count = stats.validation_confidence_count.max(1) as f64;
    println!(
        "    - radial: {:.6}",
        stats.validation_radial_sum / component_count
    );
    println!(
        "    - gravity: {:.6}",
        stats.validation_gravity_sum / component_count
    );
    println!(
        "    - coverage: {:.6}",
        stats.validation_coverage_sum / component_count
    );
    println!(
        "  - worst post-warmup confidence: {:.6}",
        stats.min_validation_confidence
    );
    println!("    - radial: {:.6}", stats.min_confidence_radial);
    println!("    - gravity: {:.6}", stats.min_confidence_gravity);
    println!("    - coverage: {:.6}", stats.min_confidence_coverage);
    println!(
        "  - post-warmup confidence range: {:.6}..={:.6}",
        stats.min_validation_confidence, stats.max_validation_confidence,
    );
    println!("- total: {} evaluations", stats.eval_count);
    println!(
        "  - until first successful correction: {} evaluations / confidence={:.6}",
        count_until_first_success,
        first_success_confidence.unwrap(),
    );
    println!("  - sampling/optimization warm-up: {warmup_count} evaluations");
    println!("  - verification: {} evaluations", stats.verified_count);

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
    let total_validation_error_sum: f64 = runs
        .iter()
        .map(|r| r.sum_validation_error_after_warmup)
        .sum();
    let total_validation_error_count: u64 = runs.iter().map(|r| r.validation_error_count).sum();
    let total_validation_confidence_sum: f64 =
        runs.iter().map(|r| r.validation_confidence_sum).sum();
    let total_validation_confidence_count: u64 =
        runs.iter().map(|r| r.validation_confidence_count).sum();
    let total_validation_radial_sum: f64 = runs.iter().map(|r| r.validation_radial_sum).sum();
    let total_validation_gravity_sum: f64 = runs.iter().map(|r| r.validation_gravity_sum).sum();
    let total_validation_coverage_sum: f64 = runs.iter().map(|r| r.validation_coverage_sum).sum();
    let worst_error_degrees: f32 = runs.iter().map(|r| r.worst_error).fold(0.0, f32::max);
    let worst_validation_error_degrees: f32 = runs
        .iter()
        .map(|r| r.worst_validation_error_after_warmup)
        .fold(0.0, f32::max);
    let min_validation_confidence = runs
        .iter()
        .map(|r| r.min_validation_confidence)
        .fold(f32::INFINITY, f32::min);
    let max_validation_confidence = runs
        .iter()
        .map(|r| r.max_validation_confidence)
        .fold(0.0, f32::max);

    println!("  ======================================================================  ");
    println!("# Average stats over {} runs", runs.len());
    println!("- evaluate_correct");
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
        "    - post-warmup: {:.3} deg over {} calls",
        total_validation_error_sum / total_validation_error_count as f64,
        avg_count(|r| r.validation_error_count),
    );
    println!("  - worst error: {worst_error_degrees:.3} deg");
    println!("    - post-warmup: {worst_validation_error_degrees:.3} deg");
    println!(
        "  - avg post-warmup confidence: {:.6} over {} calls",
        total_validation_confidence_sum / total_validation_confidence_count as f64,
        avg_count(|r| r.validation_confidence_count),
    );
    let total_count = total_validation_confidence_count.max(1) as f64;
    println!(
        "    - radial: {:.6}",
        total_validation_radial_sum / total_count
    );
    println!(
        "    - gravity: {:.6}",
        total_validation_gravity_sum / total_count
    );
    println!(
        "    - coverage: {:.6}",
        total_validation_coverage_sum / total_count
    );
    let worst_run = runs
        .iter()
        .min_by(|a, b| {
            a.min_validation_confidence
                .total_cmp(&b.min_validation_confidence)
        })
        .expect("no runs");
    println!(
        "  - worst post-warmup confidence: {:.6}",
        worst_run.min_validation_confidence
    );
    println!("    - radial: {:.6}", worst_run.min_confidence_radial);
    println!("    - gravity: {:.6}", worst_run.min_confidence_gravity);
    println!("    - coverage: {:.6}", worst_run.min_confidence_coverage);
    println!(
        "  - post-warmup confidence range: {min_validation_confidence:.6}..={max_validation_confidence:.6}"
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
        "  - sampling/optimization warm-up: {} evaluations",
        avg_count(|r| r.warmup_count),
    );
    println!(
        "  - verification: {} evaluations",
        avg_count(|r| r.verified_count),
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
