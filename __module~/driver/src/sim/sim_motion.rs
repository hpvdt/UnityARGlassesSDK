//! Deterministic simulated AR glasses for integration tests.
//!
//! Events are precomputed by a background generator thread into a small
//! bounded buffer, so [`ARGlasses::read_event`] only loads from that buffer
//! while the simulation cost stays off the caller's critical path.

use std::f32::consts::PI;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::{ARGlasses, DisplayMode, Error, GlassesEvent, Result, Side};

/// The physical gravity direction for a level stationary fixture in RUB.
pub static GRAVITY_DOWN: Vector3<f32> = Vector3::new(0.0, -9.81, 0.0);

/// The felt acceleration direction for a level stationary fixture in RUB.
pub static GRAVITY_UP: Vector3<f32> = Vector3::new(0.0, 9.81, 0.0);

/// A zero sensor vector.
pub static ZERO: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);

const DEFAULT_SEED: u64 = 0xA15E_D00D_5EED_0001;
const MAX_MAGNETIC_DIP_RAD: f32 = PI / 6.0;
const SECONDS_PER_MINUTE: f32 = 60.0;
const MICROS_PER_SECOND: f32 = 1_000_000.0;
const ANGULAR_RATE_SEGMENT_US: u64 = 10_000_000;
const ANGULAR_RATE_SEGMENT_COUNT: usize = 3;
const ADAPTIVE_CALIBRATION_HARD_IRON_DRIFT: Vector3<f32> = Vector3::new(2.0, 1.5, 2.5);
const ADAPTIVE_CALIBRATION_DRIFT_PERIOD_US: u64 = 5 * 60 * 1_000_000;
const DEFAULT_MAGNETOMETER_RANGE: f32 = 2_000.0;
/// Upper bound for the wall-clock pacing delay applied per event. Events with
/// a longer virtual period are emitted immediately.
const MAX_EVENT_PACING_DELAY: Duration = Duration::from_millis(20);

/// How many computed events the background generator may queue ahead of
/// `read_event`, bounding both memory and generator CPU usage.
const GENERATOR_BUFFERED_EVENTS: usize = 4;

/// Configuration for [`SimMotion`].
#[derive(Clone, Debug)]
pub struct Config {
    /// Seed for all deterministic random sampling.
    pub seed: u64,
    /// Virtual elapsed time between emitted sensor events, in microseconds.
    pub event_period_us: u64,
    /// Maximum absolute angular body rate per RUB axis, in rotations per minute.
    pub max_body_rate_rpm: f32,
    /// Standard deviation for linear jerk sampling, in m/s^3.
    pub linear_jerk_std_dev: f32,
    /// Exponential damping applied to linear acceleration, in 1/s.
    pub linear_accel_damping: f32,
    /// Exponential damping applied to linear velocity, in 1/s.
    pub linear_velocity_damping: f32,
    /// Maximum magnitude of simulated linear acceleration, in m/s^2.
    pub max_linear_accel: f32,
    /// Maximum magnitude of simulated linear velocity, in m/s.
    pub max_linear_velocity: f32,
    /// Gyroscope Gaussian noise standard deviation, in rad/s.
    pub gyro_noise_std_dev: f32,
    /// Accelerometer Gaussian noise standard deviation, in m/s^2.
    pub acc_noise_std_dev: f32,
    /// Magnetometer Gaussian noise standard deviation, in microtesla.
    pub mag_noise_std_dev: f32,
    /// Symmetric per-axis magnetometer range, in microtesla.
    pub magnetometer_range: f32,
    /// Magnetic field magnitude, in microtesla.
    pub magnetic_field_strength: f32,
    /// Magnetic field vertical dip from the horizon, clamped to +/-30 degrees.
    pub magnetic_dip_rad: f32,
    /// Base hard-iron magnetometer offset, in microtesla.
    pub hard_iron_base: Vector3<f32>,
    /// Slow hard-iron drift amplitude, in microtesla. Defaults to zero.
    pub hard_iron_drift: Vector3<f32>,
    /// Lower bound for every soft-iron matrix eigenvalue.
    pub soft_iron_min_eigenvalue: f32,
    /// Upper bound for every soft-iron matrix eigenvalue.
    pub soft_iron_max_eigenvalue: f32,
    /// Period for one full hard-iron drift cycle, in microseconds.
    ///
    /// This has no observable effect when [`Self::hard_iron_drift`] is zero.
    pub hard_iron_drift_period_us: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            event_period_us: 20_000,
            max_body_rate_rpm: 20.0,
            linear_jerk_std_dev: 1.2,
            linear_accel_damping: 0.8,
            linear_velocity_damping: 0.25,
            max_linear_accel: 3.0,
            max_linear_velocity: 2.0,
            gyro_noise_std_dev: 0.002,
            acc_noise_std_dev: 0.12,
            mag_noise_std_dev: 1.5,
            magnetometer_range: DEFAULT_MAGNETOMETER_RANGE,
            magnetic_field_strength: 50.0,
            magnetic_dip_rad: 20.0f32.to_radians(),
            hard_iron_base: Vector3::new(24.0, -18.0, 12.0),
            hard_iron_drift: ZERO,
            soft_iron_min_eigenvalue: 0.70,
            soft_iron_max_eigenvalue: 1.40,
            hard_iron_drift_period_us: ADAPTIVE_CALIBRATION_DRIFT_PERIOD_US,
        }
    }
}

impl Config {
    /// Returns the deterministic adaptive-calibration stress profile.
    ///
    /// Unlike the stationary default profile, this varies the hard-iron bias
    /// continuously with per-axis amplitudes of `(2.0, 1.5, 2.5)` microtesla
    /// over a five-minute cycle. All other settings, including the seed, are
    /// inherited from [`Self::default`].
    pub fn adaptive_calibration_stress() -> Self {
        Self {
            hard_iron_drift: ADAPTIVE_CALIBRATION_HARD_IRON_DRIFT,
            hard_iron_drift_period_us: ADAPTIVE_CALIBRATION_DRIFT_PERIOD_US,
            ..Self::default()
        }
    }
}

/// Current internal state of a [`SimMotion`] fixture.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Current virtual time, in microseconds.
    pub timestamp_us: u64,
    /// Whether the next emitted event will be `GlassesEvent::AccGyro`.
    pub next_event_is_acc_gyro: bool,
    /// Current body-to-world attitude in RUB coordinates.
    pub attitude: UnitQuaternion<f32>,
    /// Current world-frame position in RUB coordinates, in meters.
    pub position: Vector3<f32>,
    /// Current world-frame velocity in RUB coordinates, in m/s.
    pub velocity: Vector3<f32>,
    /// Current world-frame linear acceleration in RUB coordinates, in m/s^2.
    pub acceleration: Vector3<f32>,
    /// Most recent world-frame jerk sample in RUB coordinates, in m/s^3.
    pub jerk: Vector3<f32>,
    /// Current RUB angular body rate used by the simulator, in rad/s.
    pub angular_rate_rub: Vector3<f32>,
    /// Current display mode stored by the fixture.
    pub display_mode: DisplayMode,
    /// Current hard-iron magnetometer offset, in microtesla.
    pub hard_iron: Vector3<f32>,
    /// Current soft-iron magnetometer distortion matrix.
    pub soft_iron: Matrix3<f32>,
}

/// Public deterministic AR glasses simulator.
///
/// The fixture owns a synchronous mirror of the simulation state for
/// [`Self::snapshot`] and [`Self::accelerometer_reading`], while a background
/// generator thread runs an identically seeded copy to precompute the event
/// stream into a bounded buffer.
pub struct SimMotion {
    config: Config,
    rng: StdRng,
    attitude: UnitQuaternion<f32>,
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
    jerk: Vector3<f32>,
    angular_rate_rub: Vector3<f32>,
    angular_rate_schedule: [Vector3<f32>; ANGULAR_RATE_SEGMENT_COUNT],
    soft_iron: Matrix3<f32>,
    timestamp_us: u64,
    next_event_is_acc_gyro: bool,
    display_mode: DisplayMode,
    receiver: Receiver<GlassesEvent>,
    last_event_at: Option<Instant>,
}

impl SimMotion {
    /// Creates a deterministic SimMotion fixture with default configuration.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Creates a deterministic SimMotion fixture using `seed` and default dynamics.
    pub fn with_seed(seed: u64) -> Self {
        Self::with_config(Config {
            seed,
            ..Config::default()
        })
    }

    /// Creates a deterministic SimMotion fixture using the provided configuration.
    ///
    /// Spawns a background generator thread that computes sensor events ahead
    /// of the reader into a bounded buffer, so [`ARGlasses::read_event`] spends
    /// only the time needed to load one event from that buffer.
    pub fn with_config(config: Config) -> Self {
        let config = normalize_config(config);
        let (sender, receiver) = mpsc::sync_channel(GENERATOR_BUFFERED_EVENTS);

        // The generator copy never consumes events, so it gets a dead receiver.
        let (_, dead_receiver) = mpsc::channel();
        let mut generator = Self::build(config.clone(), dead_receiver);
        thread::spawn(move || {
            while let Ok(event) = generator.next_event() {
                if sender.send(event).is_err() {
                    break;
                }
            }
        });

        Self::build(config, receiver)
    }

    fn build(config: Config, receiver: Receiver<GlassesEvent>) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let angular_rate_schedule =
            sample_angular_rate_schedule(&mut rng, config.max_body_rate_rpm);
        let angular_rate_rub = angular_rate_schedule[0];
        let soft_iron_min_vector_norm = config.soft_iron_min_eigenvalue.sqrt();
        let soft_iron_max_vector_norm = config.soft_iron_max_eigenvalue.sqrt();
        let soft_iron = sample_soft_iron(
            &mut rng,
            soft_iron_min_vector_norm,
            soft_iron_max_vector_norm,
        );

        Self {
            config,
            rng,
            attitude: UnitQuaternion::identity(),
            position: ZERO,
            velocity: ZERO,
            acceleration: ZERO,
            jerk: ZERO,
            angular_rate_rub,
            angular_rate_schedule,
            soft_iron,
            timestamp_us: 0,
            next_event_is_acc_gyro: true,
            display_mode: DisplayMode::SameOnBoth,
            receiver,
            last_event_at: None,
        }
    }

    /// Returns a copy of the fixture's current internal state.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            timestamp_us: self.timestamp_us,
            next_event_is_acc_gyro: self.next_event_is_acc_gyro,
            attitude: self.attitude,
            position: self.position,
            velocity: self.velocity,
            acceleration: self.acceleration,
            jerk: self.jerk,
            angular_rate_rub: self.angular_rate_rub,
            display_mode: self.display_mode,
            hard_iron: self.current_hard_iron(),
            soft_iron: self.current_soft_iron(),
        }
    }

    /// Computes the next event synchronously: paces emissions to the
    /// configured event period in wall time, samples the sensors, and advances
    /// the simulation. Used by the background generator thread; the public
    /// [`ARGlasses::read_event`] loads precomputed events from the buffer.
    fn next_event(&mut self) -> Result<GlassesEvent> {
        if let Some(last) = self.last_event_at {
            let period = Duration::from_micros(self.config.event_period_us);
            if period <= MAX_EVENT_PACING_DELAY {
                let delay = period.saturating_sub(last.elapsed());
                if !delay.is_zero() {
                    thread::sleep(delay);
                }
            }
        }
        self.last_event_at = Some(Instant::now());

        let timestamp = self.timestamp_us;
        let result = if self.next_event_is_acc_gyro {
            self.next_event_is_acc_gyro = false;
            GlassesEvent::AccGyro {
                accelerometer: self.accelerometer_reading(),
                gyroscope: self.gyroscope_reading(),
                timestamp,
            }
        } else {
            self.next_event_is_acc_gyro = true;
            GlassesEvent::Magnetometer {
                magnetometer: self.magnetometer_reading(),
                timestamp,
            }
        };

        self.advance_state();

        Ok(result)
    }

    fn advance_state(&mut self) {
        let dt = self.config.event_period_us as f32 / MICROS_PER_SECOND;
        self.timestamp_us = self
            .timestamp_us
            .saturating_add(self.config.event_period_us);

        let rotation_increment = UnitQuaternion::from_scaled_axis(self.angular_rate_rub * dt);
        self.attitude *= rotation_increment;
        self.attitude.renormalize();

        let segment =
            (self.timestamp_us / ANGULAR_RATE_SEGMENT_US) as usize % ANGULAR_RATE_SEGMENT_COUNT;
        self.angular_rate_rub = self.angular_rate_schedule[segment];

        self.jerk = self.sample_noise_vec(self.config.linear_jerk_std_dev);
        self.acceleration += self.jerk * dt;
        self.acceleration *= damping_factor(self.config.linear_accel_damping, dt);
        self.acceleration = clamp_norm(self.acceleration, self.config.max_linear_accel);

        self.velocity += self.acceleration * dt;
        self.velocity *= damping_factor(self.config.linear_velocity_damping, dt);
        self.velocity = clamp_norm(self.velocity, self.config.max_linear_velocity);

        self.position += self.velocity * dt;
    }

    /// Samples the noisy accelerometer at the current state without advancing
    /// virtual time. Repeated calls consume new sensor noise samples.
    pub fn accelerometer_reading(&mut self) -> Vector3<f32> {
        let felt_acceleration = GRAVITY_UP + self.acceleration;
        let body_acceleration = self.attitude.inverse() * felt_acceleration;
        body_acceleration + self.sample_noise_vec(self.config.acc_noise_std_dev)
    }

    fn gyroscope_reading(&mut self) -> Vector3<f32> {
        self.angular_rate_rub + self.sample_noise_vec(self.config.gyro_noise_std_dev)
    }

    fn magnetometer_reading(&mut self) -> Vector3<f32> {
        let magnetic_world = self.magnetic_north_world_rub();
        let ideal_body = self.attitude.inverse() * magnetic_world;
        let distorted = self.current_soft_iron() * ideal_body + self.current_hard_iron();
        let noisy = distorted + self.sample_noise_vec(self.config.mag_noise_std_dev);

        noisy.map(|component| {
            component.clamp(
                -self.config.magnetometer_range,
                self.config.magnetometer_range,
            )
        })
    }

    fn magnetic_north_world_rub(&self) -> Vector3<f32> {
        let dip = self
            .config
            .magnetic_dip_rad
            .clamp(-MAX_MAGNETIC_DIP_RAD, MAX_MAGNETIC_DIP_RAD);
        Vector3::new(0.0, dip.sin(), -dip.cos()) * self.config.magnetic_field_strength
    }

    fn current_hard_iron(&self) -> Vector3<f32> {
        let phase = self.drift_phase();
        let drift_shape = Vector3::new(
            phase.sin(),
            (phase * 0.73 + 1.1).sin(),
            (phase * 0.37 + 2.3).sin(),
        );

        self.config.hard_iron_base + self.config.hard_iron_drift.component_mul(&drift_shape)
    }

    fn current_soft_iron(&self) -> Matrix3<f32> {
        self.soft_iron
    }

    fn drift_phase(&self) -> f32 {
        let period = self.config.hard_iron_drift_period_us as f32;
        2.0 * PI * self.timestamp_us as f32 / period
    }

    fn sample_noise_vec(&mut self, std_dev: f32) -> Vector3<f32> {
        Vector3::new(
            sample_noise(&mut self.rng, std_dev),
            sample_noise(&mut self.rng, std_dev),
            sample_noise(&mut self.rng, std_dev),
        )
    }
}

impl Default for SimMotion {
    fn default() -> Self {
        Self::new()
    }
}

impl ARGlasses for SimMotion {
    fn serial(&mut self) -> Result<String> {
        Ok(String::from("sim_motion!"))
    }

    /// Blocks briefly to pace emissions to the configured event period, then
    /// loads the next event from the buffer that the background generator
    /// thread keeps filled ahead of the reader. Simulation cost is therefore
    /// not paid on this call path. Virtual periods longer than
    /// [`MAX_EVENT_PACING_DELAY`] cannot be represented this way, so events
    /// are emitted immediately instead.
    ///
    /// Returns [`Error::Other`] if the generator thread has stopped.
    fn read_event(&mut self) -> Result<GlassesEvent> {
        if let Some(last) = self.last_event_at {
            let period = Duration::from_micros(self.config.event_period_us);
            if period <= MAX_EVENT_PACING_DELAY {
                let delay = period.saturating_sub(last.elapsed());
                if !delay.is_zero() {
                    thread::sleep(delay);
                }
            }
        }
        self.last_event_at = Some(Instant::now());

        let result = self
            .receiver
            .recv()
            .map_err(|_| Error::Other("sim_motion event generator stopped"))?;

        // Advance the mirror state once per consumed event so `snapshot()`
        // keeps matching the emitted event sequence exactly.
        self.advance_state();
        self.next_event_is_acc_gyro = !self.next_event_is_acc_gyro;

        Ok(result)
    }

    fn get_display_mode(&mut self) -> Result<DisplayMode> {
        Ok(self.display_mode)
    }

    fn set_display_mode(&mut self, display_mode: DisplayMode) -> Result<()> {
        self.display_mode = display_mode;
        Ok(())
    }

    fn display_fov(&self) -> f32 {
        24.0f32.to_radians()
    }

    fn imu_to_display_matrix(&self, _side: Side, _ipd: f32) -> Isometry3<f64> {
        Isometry3::identity()
    }

    fn name(&self) -> &'static str {
        "sim_motion"
    }

    fn display_delay(&self) -> u64 {
        0
    }
}

fn normalize_config(mut config: Config) -> Config {
    config.event_period_us = config.event_period_us.max(1);
    config.max_body_rate_rpm = config.max_body_rate_rpm.max(0.0);
    config.linear_jerk_std_dev = config.linear_jerk_std_dev.max(0.0);
    config.linear_accel_damping = config.linear_accel_damping.max(0.0);
    config.linear_velocity_damping = config.linear_velocity_damping.max(0.0);
    config.max_linear_accel = config.max_linear_accel.max(0.0);
    config.max_linear_velocity = config.max_linear_velocity.max(0.0);
    config.gyro_noise_std_dev = config.gyro_noise_std_dev.max(0.0);
    config.acc_noise_std_dev = config.acc_noise_std_dev.max(0.0);
    config.mag_noise_std_dev = config.mag_noise_std_dev.max(0.0);
    if !config.magnetometer_range.is_finite() || config.magnetometer_range <= 0.0 {
        config.magnetometer_range = DEFAULT_MAGNETOMETER_RANGE;
    }
    config.magnetic_field_strength = config.magnetic_field_strength.abs();
    config.magnetic_dip_rad = config
        .magnetic_dip_rad
        .clamp(-MAX_MAGNETIC_DIP_RAD, MAX_MAGNETIC_DIP_RAD);
    if !config.soft_iron_min_eigenvalue.is_finite() || config.soft_iron_min_eigenvalue <= 0.0 {
        config.soft_iron_min_eigenvalue = Config::default().soft_iron_min_eigenvalue;
    }
    if !config.soft_iron_max_eigenvalue.is_finite()
        || config.soft_iron_max_eigenvalue < config.soft_iron_min_eigenvalue
    {
        config.soft_iron_max_eigenvalue = config.soft_iron_min_eigenvalue;
    }
    config.hard_iron_drift_period_us = config.hard_iron_drift_period_us.max(1);

    config
}

fn sample_soft_iron(
    rng: &mut StdRng,
    minimum_vector_norm: f32,
    maximum_vector_norm: f32,
) -> Matrix3<f32> {
    let mut vectors = [ZERO; 3];

    for index in 0..vectors.len() {
        loop {
            let mut vector = Vector3::new(
                rng.gen_range(-1.0..=1.0),
                rng.gen_range(-1.0..=1.0),
                rng.gen_range(-1.0..=1.0),
            );

            for previous in &vectors[..index] {
                vector -= previous * (vector.dot(previous) / previous.norm_squared());
            }

            if vector.norm_squared() > f32::EPSILON {
                vectors[index] =
                    clamp_vector_norm(vector, minimum_vector_norm, maximum_vector_norm);
                break;
            }
        }
    }

    let scaled_eigenvectors = Matrix3::from_columns(&vectors);
    scaled_eigenvectors * scaled_eigenvectors.transpose()
}

fn clamp_vector_norm(vector: Vector3<f32>, minimum_norm: f32, maximum_norm: f32) -> Vector3<f32> {
    let norm = vector.norm();
    vector * (norm.clamp(minimum_norm, maximum_norm) / norm)
}

fn sample_angular_rate_schedule(
    rng: &mut StdRng,
    max_body_rate_rpm: f32,
) -> [Vector3<f32>; ANGULAR_RATE_SEGMENT_COUNT] {
    let max_rad_per_sec = max_body_rate_rpm * 2.0 * PI / SECONDS_PER_MINUTE;

    if max_rad_per_sec <= 0.0 {
        return [ZERO; ANGULAR_RATE_SEGMENT_COUNT];
    }

    std::array::from_fn(|dominant_axis| {
        let mut rate = Vector3::new(
            rng.gen_range(0.10..=0.30) * max_rad_per_sec,
            rng.gen_range(0.10..=0.30) * max_rad_per_sec,
            rng.gen_range(0.10..=0.30) * max_rad_per_sec,
        );
        rate[dominant_axis] = rng.gen_range(0.70..=1.0) * max_rad_per_sec;
        rate
    })
}

fn sample_noise(rng: &mut StdRng, std_dev: f32) -> f32 {
    if std_dev <= 0.0 || !std_dev.is_finite() {
        return 0.0;
    }

    Normal::new(0.0, std_dev as f64)
        .map(|normal| normal.sample(rng) as f32)
        .unwrap_or(0.0)
}

fn damping_factor(rate: f32, dt: f32) -> f32 {
    (-rate * dt).exp()
}

fn clamp_norm(v: Vector3<f32>, max_norm: f32) -> Vector3<f32> {
    let norm = v.norm();
    if max_norm > 0.0 && norm > max_norm {
        v * (max_norm / norm)
    } else {
        v
    }
}

#[cfg(test)]
#[path = "sim_motion_tests.rs"]
mod sim_motion_tests;
