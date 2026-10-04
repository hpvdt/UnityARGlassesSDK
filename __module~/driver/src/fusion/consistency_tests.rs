use super::{Consistency, ConsistencyStatus, SourceConsistency};

#[test]
fn pending_until_minimum_samples() {
    let mut source = SourceConsistency::new(0.04);
    assert_eq!(source.status(), ConsistencyStatus::Pending);
    for _ in 0..SourceConsistency::MIN_SAMPLES - 1 {
        assert_eq!(source.record(0.05), ConsistencyStatus::Pending);
    }
    assert_eq!(
        source.record(0.05),
        ConsistencyStatus::Consistent,
        "verdict appears exactly at MIN_SAMPLES"
    );
}

#[test]
fn small_innovations_report_consistent() {
    let mut source = SourceConsistency::new(0.04); // sigma = 0.2 rad
    for _ in 0..50 {
        source.record(0.05);
    }
    assert_eq!(source.status(), ConsistencyStatus::Consistent);
    // exponential average residual after 50 samples: value * 0.9^50 ~= 2.6e-4
    assert!((source.innovation.ema - 0.05).abs() < 5e-4);
    // test ratio = (0.05 / 3)^2 / 0.04 ~= 0.00694 (gate-normalized like both stacks)
    let expected_ratio = (0.05_f32 / 3.0).powi(2) / 0.04;
    assert!((source.test_ratio.ema - expected_ratio).abs() < 5e-4);
    assert_eq!(source.rejected_count, 0);
}

#[test]
fn large_innovations_report_inconsistent() {
    let mut source = SourceConsistency::new(0.04);
    for _ in 0..50 {
        source.record(0.05);
    }
    assert_eq!(source.status(), ConsistencyStatus::Consistent);

    for _ in 0..10 {
        source.record(1.0); // 5-sigma innovation: test ratio = (5/3)^2 > 1
    }
    assert_eq!(source.status(), ConsistencyStatus::Inconsistent);
    assert!(source.test_ratio.ema >= 1.0);
    assert_eq!(source.rejected_count, 10);
    assert!(source.innovation_rejected);
}

#[test]
fn outlier_counting_tracks_latest_only_for_rejection_flag() {
    let mut source = SourceConsistency::new(0.04);
    source.record(0.8); // 4 sigma at the default 3-sigma gate
    assert!(source.innovation_rejected);
    assert_eq!(source.rejected_count, 1);
    source.record(0.05);
    assert!(!source.innovation_rejected);
    assert_eq!(
        source.rejected_count, 1,
        "the earlier outlier stays counted"
    );
}

#[test]
fn innovation_gate_builder_and_floor_match_flight_stacks() {
    // floor: a zero gate is clamped to 1 sigma like ArduPilot `MAX(0.01f * gate, 1.0f)`
    let mut floored = SourceConsistency::new(0.04).innovation_gate(0.0);
    floored.record(0.5); // 2.5 sigma
    assert!(
        floored.innovation_rejected,
        "gate floored to 1 sigma, so 2.5 sigma fails"
    );

    // same sample passes at the default 3-sigma gate
    let mut default_gate = SourceConsistency::new(0.04);
    default_gate.record(0.5);
    assert!(!default_gate.innovation_rejected);
}

#[test]
fn record_scaled_recovers_pre_correction_innovation() {
    let mut source = SourceConsistency::new(0.04);
    let status = source.record_scaled(0.005, 0.05);
    assert!(status.is_some());
    assert!(
        (source.innovation.last - 0.1).abs() < 1e-6,
        "scaled=0.005 with blend ratio 0.05 reconstructs innovation 0.1, got {}",
        source.innovation.last
    );
}

#[test]
fn record_scaled_ignores_disabled_blend() {
    let mut source = SourceConsistency::new(0.04);
    assert_eq!(source.record_scaled(0.005, 0.0), None);
    assert_eq!(
        source.samples_count, 0,
        "a zero blend ratio records nothing"
    );
}

#[test]
fn record_with_variance_normalizes_by_live_variance() {
    let mut source = SourceConsistency::new(0.04);
    let status = source.record_with_variance(1.0, 1.0);
    assert!(status.is_some());
    assert!(
        (source.test_ratio.last - 1.0 / 9.0).abs() < 1e-6,
        "innovation equals live sigma, so test ratio is 1/gate^2 = 1/9, got {}",
        source.test_ratio.last
    );
    assert!(
        (source.innovation_variance.ema - 0.136).abs() < 1e-6,
        "stored variance tracks the live value: 0.04 * 0.9 + 1.0 * 0.1, got {}",
        source.innovation_variance.ema
    );
}

#[test]
fn record_with_variance_rejects_degenerate_variance() {
    let mut source = SourceConsistency::new(0.04);
    assert_eq!(source.record_with_variance(0.1, 0.0), None);
    assert_eq!(source.record_with_variance(0.1, -1.0), None);
    assert_eq!(source.record_with_variance(0.1, f32::NAN), None);
    assert_eq!(source.samples_count, 0);
}

#[test]
fn aggregate_reports_worst_source() {
    let mut consistency = Consistency::attitude_defaults();
    for _ in 0..10 {
        consistency.sources.acc.record(0.05);
        consistency.sources.gyro.record(0.02);
        consistency.sources.mag.record(0.1);
    }
    assert_eq!(consistency.status(), ConsistencyStatus::Consistent);
    assert!(consistency.worst_test_ratio() < 1.0);

    for _ in 0..10 {
        consistency.sources.acc.record(0.05);
        consistency.sources.gyro.record(0.02);
        consistency.sources.mag.record(5.0); // 10 sigma heading disagreement
    }
    assert_eq!(consistency.status(), ConsistencyStatus::Inconsistent);
    assert!(
        consistency.worst_test_ratio() >= 1.0,
        "the magnetometer sinks the aggregate, got {}",
        consistency.worst_test_ratio()
    );
}

#[test]
fn aggregate_stays_pending_while_any_source_pending() {
    let mut consistency = Consistency::attitude_defaults();
    for _ in 0..10 {
        consistency.sources.acc.record(0.05);
        consistency.sources.gyro.record(0.02);
    }
    assert_eq!(consistency.status(), ConsistencyStatus::Pending);
}
