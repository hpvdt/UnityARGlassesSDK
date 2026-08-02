use super::*;
use crate::fusion::rub_to_frd;

fn quiet_config() -> Config {
    Config {
        max_body_rate_rpm: 0.0,
        linear_jerk_std_dev: 0.0,
        gyro_noise_std_dev: 0.0,
        acc_noise_std_dev: 0.0,
        mag_noise_std_dev: 0.0,
        magnetic_dip_rad: 0.0,
        hard_iron_base: ZERO,
        hard_iron_drift: ZERO,
        soft_iron_min_eigenvalue: 1.0,
        soft_iron_max_eigenvalue: 1.0,
        ..Config::default()
    }
}

fn moving_config() -> Config {
    Config {
        event_period_us: 100_000,
        max_body_rate_rpm: 1.0,
        ..quiet_config()
    }
}

#[test]
fn starts_level_with_felt_gravity_up_frd() {
    let mut sim_motion = SimMotion::with_config(quiet_config());

    match sim_motion.read_event().unwrap() {
        GlassesEvent::AccGyro {
            accelerometer,
            timestamp,
            ..
        } => {
            assert_eq!(timestamp, 0);
            let acc_frd = rub_to_frd(&accelerometer);
            let expected = Vector3::new(0.0, 0.0, -9.81);

            assert!(
                (acc_frd - expected).norm() < 1.0e-5,
                "acc_frd={:?}",
                acc_frd
            );
        }
        event => panic!("expected AccGyro, got {:?}", event),
    }
}

#[test]
fn starts_with_magnetic_north_forward_frd() {
    let mut sim_motion = SimMotion::with_config(quiet_config());
    let _ = sim_motion.read_event().unwrap();

    match sim_motion.read_event().unwrap() {
        GlassesEvent::Magnetometer {
            magnetometer,
            timestamp,
        } => {
            assert_eq!(timestamp, Config::default().event_period_us);
            let mag_frd = rub_to_frd(&magnetometer);
            let expected = Vector3::new(50.0, 0.0, 0.0);

            assert!(
                (mag_frd - expected).norm() < 1.0e-5,
                "mag_frd={:?}",
                mag_frd
            );
        }
        event => panic!("expected Magnetometer, got {:?}", event),
    }
}

#[test]
fn default_hard_iron_bias_is_under_50_microtesla() {
    let hard_iron = SimMotion::new().snapshot().hard_iron;

    assert!(
        hard_iron.norm() < 50.0,
        "hard_iron={:?}, norm={}",
        hard_iron,
        hard_iron.norm()
    );
}

#[test]
fn default_body_rate_is_bounded_at_twenty_rpm_per_axis() {
    let config = Config::default();
    let max_rad_per_sec = config.max_body_rate_rpm * 2.0 * PI / SECONDS_PER_MINUTE;
    let angular_rate = SimMotion::new().snapshot().angular_rate_rub;

    assert_eq!(config.max_body_rate_rpm, 20.0);
    for component in angular_rate.iter() {
        assert!(*component > 0.0 && *component <= max_rad_per_sec);
    }
}

#[test]
fn default_motion_changes_ideal_magnetic_signal_more_than_configured_noise() {
    let default_config = Config::default();
    let configured_noise = default_config.mag_noise_std_dev;
    let mut sim_motion = SimMotion::with_config(Config {
        linear_jerk_std_dev: 0.0,
        mag_noise_std_dev: 0.0,
        hard_iron_base: ZERO,
        hard_iron_drift: ZERO,
        soft_iron_min_eigenvalue: 1.0,
        soft_iron_max_eigenvalue: 1.0,
        ..default_config
    });
    let mut magnetometers = Vec::new();

    while magnetometers.len() < 2 {
        if let GlassesEvent::Magnetometer { magnetometer, .. } = sim_motion.read_event().unwrap() {
            magnetometers.push(magnetometer);
        }
    }

    let ideal_signal_change = (magnetometers[1] - magnetometers[0]).norm();
    assert!(
        ideal_signal_change > configured_noise,
        "ideal_signal_change={}, configured_noise={}",
        ideal_signal_change,
        configured_noise
    );
}

#[test]
fn soft_iron_is_fixed_positive_definite_and_bounded() {
    let mut sim_motion = SimMotion::new();
    let config = sim_motion.config.clone();
    let lower_eigenvalue = config.soft_iron_min_eigenvalue;
    let upper_eigenvalue = config.soft_iron_max_eigenvalue;
    let initial_soft_iron = sim_motion.snapshot().soft_iron;

    for sample in 0..=360 {
        sim_motion.timestamp_us = config.hard_iron_drift_period_us * sample / 360;
        let soft_iron = sim_motion.snapshot().soft_iron;
        let asymmetry = soft_iron - soft_iron.transpose();

        assert_eq!(soft_iron, initial_soft_iron);
        assert!(asymmetry.norm() <= 1.0e-5, "soft_iron={:?}", soft_iron);
        assert!(soft_iron.cholesky().is_some(), "soft_iron={:?}", soft_iron);

        for eigenvalue in soft_iron.symmetric_eigen().eigenvalues.iter() {
            assert!(
                *eigenvalue >= lower_eigenvalue - 1.0e-4
                    && *eigenvalue <= upper_eigenvalue + 1.0e-4,
                "eigenvalue={}, soft_iron={:?}",
                eigenvalue,
                soft_iron
            );
        }
    }
}

#[test]
fn gyro_matches_each_integrated_angular_rate() {
    let config = moving_config();
    let dt = config.event_period_us as f32 / MICROS_PER_SECOND;
    let max_rad_per_sec = config.max_body_rate_rpm * 2.0 * PI / SECONDS_PER_MINUTE;
    let mut sim_motion = SimMotion::with_config(config);
    let mut observed_rates = Vec::new();

    for _ in 0..350 {
        let before = sim_motion.snapshot();
        let event = sim_motion.read_event().unwrap();
        let after = sim_motion.snapshot();
        let expected_attitude =
            before.attitude * UnitQuaternion::from_scaled_axis(before.angular_rate_rub * dt);

        assert!(
            expected_attitude.angle_to(&after.attitude) < 1.0e-5,
            "before={:?}, rate={:?}, after={:?}",
            before.attitude,
            before.angular_rate_rub,
            after.attitude
        );

        if let GlassesEvent::AccGyro { gyroscope, .. } = event {
            assert!((gyroscope - before.angular_rate_rub).norm() < 1.0e-7);
            observed_rates.push(gyroscope);
        }

        for component in before.angular_rate_rub.iter() {
            assert!(*component > 0.0 && *component <= max_rad_per_sec);
        }
    }

    assert!(
        observed_rates
            .windows(2)
            .any(|rates| rates[0].cross(&rates[1]).norm() > 1.0e-4),
        "observed_rates={:?}",
        observed_rates
    );
}

#[test]
fn trajectory_covers_roll_pitch_yaw_and_non_planar_magnetometer_space() {
    let mut sim_motion = SimMotion::with_config(moving_config());
    let mut angle_min = Vector3::repeat(f32::INFINITY);
    let mut angle_max = Vector3::repeat(f32::NEG_INFINITY);
    let mut magnetometers = Vec::new();

    for _ in 0..3_600 {
        if let GlassesEvent::Magnetometer { magnetometer, .. } = sim_motion.read_event().unwrap() {
            magnetometers.push(magnetometer);
            let (roll, pitch, yaw) = sim_motion.snapshot().attitude.euler_angles();
            let angles = Vector3::new(roll, pitch, yaw);
            angle_min = angle_min.zip_map(&angles, f32::min);
            angle_max = angle_max.zip_map(&angles, f32::max);
        }
    }

    let angle_range = angle_max - angle_min;
    for range in angle_range.iter() {
        assert!(*range > 1.0, "angle_range={:?}", angle_range);
    }

    let mean = magnetometers.iter().copied().sum::<Vector3<f32>>() / magnetometers.len() as f32;
    let covariance = magnetometers.iter().fold(Matrix3::zeros(), |sum, sample| {
        let centered = sample - mean;
        sum + centered * centered.transpose()
    }) / magnetometers.len() as f32;
    let eigenvalues = covariance.symmetric_eigen().eigenvalues;
    let minimum = eigenvalues.min();
    let maximum = eigenvalues.max();

    assert!(
        minimum / maximum > 0.03,
        "eigenvalues={:?}, covariance={:?}",
        eigenvalues,
        covariance
    );
}
