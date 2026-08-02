use ar_drivers::sim::sim_motion::Config;
use ar_drivers::{ARGlasses, GlassesEvent, SimMotion};
use nalgebra::Vector3;

const DEFAULT_MAGNETOMETER_RANGE: f32 = 2_000.0;

fn magnetometer_for_range(range: f32, hard_iron: Vector3<f32>) -> Vector3<f32> {
    let mut sim_motion = SimMotion::with_config(Config {
        max_body_rate_rpm: 0.0,
        linear_jerk_std_dev: 0.0,
        mag_noise_std_dev: 0.0,
        magnetometer_range: range,
        magnetic_field_strength: 0.0,
        hard_iron_base: hard_iron,
        hard_iron_drift: Vector3::zeros(),
        soft_iron_min_eigenvalue: 1.0,
        soft_iron_max_eigenvalue: 1.0,
        ..Config::default()
    });

    sim_motion.read_event().unwrap();
    match sim_motion.read_event().unwrap() {
        GlassesEvent::Magnetometer { magnetometer, .. } => magnetometer,
        event => panic!("expected Magnetometer, got {:?}", event),
    }
}

#[test]
fn configured_range_saturates_positive_and_negative_components() {
    let magnetometer = magnetometer_for_range(10.0, Vector3::new(12.0, -12.0, 4.0));

    assert_eq!(magnetometer, Vector3::new(10.0, -10.0, 4.0));
}

#[test]
fn default_range_is_finite_and_symmetric() {
    assert_eq!(
        Config::default().magnetometer_range,
        DEFAULT_MAGNETOMETER_RANGE
    );
    assert_eq!(
        magnetometer_for_range(
            Config::default().magnetometer_range,
            Vector3::new(3_000.0, -3_000.0, 0.0),
        ),
        Vector3::new(DEFAULT_MAGNETOMETER_RANGE, -DEFAULT_MAGNETOMETER_RANGE, 0.0,)
    );
}

#[test]
fn invalid_ranges_fall_back_to_default() {
    for range in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let magnetometer = magnetometer_for_range(range, Vector3::new(3_000.0, -3_000.0, 0.0));

        assert_eq!(
            magnetometer,
            Vector3::new(DEFAULT_MAGNETOMETER_RANGE, -DEFAULT_MAGNETOMETER_RANGE, 0.0,),
            "range={range}"
        );
    }
}

#[test]
fn minimum_positive_range_is_preserved() {
    let range = f32::MIN_POSITIVE;
    let magnetometer = magnetometer_for_range(range, Vector3::new(range * 2.0, -range * 2.0, 0.0));

    assert_eq!(magnetometer, Vector3::new(range, -range, 0.0));
}
