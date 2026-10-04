use super::{Consistency, ConsistencyStatus, SourceConsistency};

#[test]
fn pending_until_minimum_samples() {
    let mut source = SourceConsistency::new(0.2);
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
    let mut source = SourceConsistency::new(0.2);
    for _ in 0..50 {
        source.record(0.05);
    }
    assert_eq!(source.status(), ConsistencyStatus::Consistent);
    // exponential average residual after 50 samples: value * 0.9^50 ~= 2.6e-4
    assert!((source.avg - 0.05).abs() < 5e-4);
    assert!((source.avg_test_ratio - 0.0625).abs() < 5e-4);
    assert_eq!(source.rejected, 0);
}

#[test]
fn large_innovations_report_inconsistent() {
    let mut source = SourceConsistency::new(0.2);
    for _ in 0..50 {
        source.record(0.05);
    }
    assert_eq!(source.status(), ConsistencyStatus::Consistent);

    for _ in 0..10 {
        source.record(1.0); // 5-sigma innovation
    }
    assert_eq!(source.status(), ConsistencyStatus::Inconsistent);
    assert!(source.avg_test_ratio >= 1.0);
    assert_eq!(source.rejected, 10);
}

#[test]
fn outlier_counting_tracks_latest_only_for_is_rejected() {
    let mut source = SourceConsistency::new(0.2);
    source.record(0.8); // 4 sigma
    assert!(source.is_rejected());
    assert_eq!(source.rejected, 1);
    source.record(0.05);
    assert!(!source.is_rejected());
    assert_eq!(source.rejected, 1, "the earlier outlier stays counted");
}

#[test]
fn record_scaled_recovers_pre_correction_innovation() {
    let mut source = SourceConsistency::new(0.2);
    let status = source.record_scaled(0.005, 0.05);
    assert!(status.is_some());
    assert!(
        (source.latest - 0.1).abs() < 1e-6,
        "scaled=0.005 with blend ratio 0.05 reconstructs innovation 0.1, got {}",
        source.latest
    );
}

#[test]
fn record_scaled_ignores_disabled_blend() {
    let mut source = SourceConsistency::new(0.2);
    assert_eq!(source.record_scaled(0.005, 0.0), None);
    assert_eq!(source.samples, 0, "a zero blend ratio records nothing");
}

#[test]
fn record_with_variance_normalizes_by_live_gate() {
    let mut source = SourceConsistency::new(0.2);
    let status = source.record_with_variance(1.0, 1.0);
    assert!(status.is_some());
    assert!(
        (source.test_ratio - 1.0).abs() < 1e-6,
        "innovation equals live gate, test ratio is 1"
    );
    assert!(
        (source.gate - 0.28).abs() < 1e-6,
        "stored gate tracks the live spread: 0.2 * 0.9 + 1.0 * 0.1, got {}",
        source.gate
    );
}

#[test]
fn record_with_variance_rejects_degenerate_variance() {
    let mut source = SourceConsistency::new(0.2);
    assert_eq!(source.record_with_variance(0.1, 0.0), None);
    assert_eq!(source.record_with_variance(0.1, -1.0), None);
    assert_eq!(source.record_with_variance(0.1, f32::NAN), None);
    assert_eq!(source.samples, 0);
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
