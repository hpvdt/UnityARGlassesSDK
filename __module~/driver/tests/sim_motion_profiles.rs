use ar_drivers::sim::sim_motion::Config;
use ar_drivers::{ARGlasses, SimMotion};
use nalgebra::Vector3;

const FIVE_MINUTES_US: u64 = 5 * 60 * 1_000_000;

#[test]
fn default_hard_iron_bias_is_stationary() {
    let mut config = Config::default();

    assert_eq!(config.hard_iron_drift, Vector3::zeros());
    config.event_period_us = config.hard_iron_drift_period_us / 12;

    let mut sim_motion = SimMotion::with_config(config);
    let initial_hard_iron = sim_motion.snapshot().hard_iron;

    for _ in 0..12 {
        sim_motion.read_event().unwrap();
        assert_eq!(sim_motion.snapshot().hard_iron, initial_hard_iron);
    }
}

#[test]
fn adaptive_calibration_stress_profile_drifts_deterministically() {
    let mut config = Config::adaptive_calibration_stress();

    assert_eq!(config.hard_iron_drift, Vector3::new(2.0, 1.5, 2.5));
    assert_eq!(config.hard_iron_drift_period_us, FIVE_MINUTES_US);
    config.event_period_us = config.hard_iron_drift_period_us / 12;

    let mut first = SimMotion::with_config(config.clone());
    let mut second = SimMotion::with_config(config);
    let initial_hard_iron = first.snapshot().hard_iron;
    let mut observed_drift = false;

    for _ in 0..12 {
        first.read_event().unwrap();
        second.read_event().unwrap();
        let first_hard_iron = first.snapshot().hard_iron;
        let second_hard_iron = second.snapshot().hard_iron;

        assert_eq!(first_hard_iron, second_hard_iron);
        observed_drift |= (first_hard_iron - initial_hard_iron).norm() > 1.0e-3;
    }

    assert!(observed_drift);
}
