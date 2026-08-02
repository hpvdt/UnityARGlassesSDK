## High severity

- [x] Generate a non-planar attitude trajectory

    - **Summary:** A constant three-component angular-rate vector still rotates about one fixed axis, so the simulated
      magnetometer samples do not cover a three-dimensional ellipsoid.
    - **Affected module:** `src/sim/sim_motion.rs`
    - **Severity:** High
    - **Description:** SimMotion samples one angular-rate vector during construction and reuses it forever:

      ```rust
      let angular_rate_rub = sample_angular_rate(&mut rng, config.max_body_rate_rpm);
  
      let rotation_increment = UnitQuaternion::from_scaled_axis(self.angular_rate_rub * dt);
      self.attitude *= rotation_increment;
      ```

      Rotation around a fixed axis makes the ideal body-frame magnetic vector trace a planar circle, and a fixed affine
      distortion maps that circle to another plane. Noise can make the calibration design appear full-rank, but it does
      not supply the missing orientation coverage and can instead produce a non-physical fitted conic.
    - **Recommended fix:** Change the body-rate direction over time with a seeded multi-axis excitation schedule that
      visits substantially different roll, pitch, and yaw orientations.

## Medium severity

- [x] Balance default angular motion against magnetometer noise

    - **Summary:** The default SimMotion changes the noiseless magnetic vector much more slowly than it perturbs each
      measurement, so early apparent diversity is dominated by noise rather than orientation.
    - **Affected module:** `src/sim/sim_motion.rs`
    - **Severity:** Medium
    - **Description:** Defaults combine a maximum of 1 RPM per body axis, 20 ms between magnetometer samples, a 50
      microtesla field, and 1.5 microtesla per-component noise:

      ```rust
      event_period_us: 10_000,
      max_body_rate_rpm: 1.0,
      mag_noise_std_dev: 1.5,
      magnetic_field_strength: 50.0,
      ```

      Even at the maximum combined rate, the ideal field changes by only about 0.18 microtesla between magnetometer
      samples, while the standard deviation of the difference between two independent noisy samples is about 2.1
      microtesla per component. For the measured default seed, the ideal change is limited to about 0.10 microtesla per
      magnetometer sample.
    - **Recommended fix:** Set the default maximum body rate to be 10 ~ 20 RPM per body axis.

- [x] Separate static-calibration and drifting-bias simulator profiles

    - **Summary:** The default simulator continuously changes its hard-iron offset, so it does not provide a stationary
      ellipsoid for validating a static calibration solve.
    - **Affected module:** `src/sim/sim_motion.rs`
    - **Severity:** Medium
    - **Description:** The default has a five-minute drift cycle that changes each bias component with a different phase
      and frequency multiplier:

      ```rust
      hard_iron_drift: Vector3::new(2.0, 1.5, 2.5),
      hard_iron_drift_period_us: 5 * 60 * 1_000_000,
  
      self.config.hard_iron_base
          + self.config.hard_iron_drift.component_mul(&drift_shape)
      ```

      A static ellipsoid solver and an adaptive-bias stress simulation test different behaviors, but the default SimMotion
      currently combines them in the same deterministic stream.
    - **Recommended fix:** Make the baseline default use zero hard-iron drift and retain the current drift
      as a separately selected adaptive-calibration stress profile.

- [x] Model finite magnetometer range instead of unbounded output

    - **Summary:** The simulator adds unbounded Gaussian noise without applying a sensor range, so a sufficiently long
      immediate stream can emit arbitrarily extreme but finite readings.
    - **Affected module:** `src/sim/sim_motion.rs`
    - **Severity:** Medium
    - **Description:** Noise is sampled independently and added directly to the distorted magnetic field:

      ```rust
      distorted + self.sample_noise_vec(self.config.mag_noise_std_dev)
  
      Normal::new(0.0, std_dev as f64)
          .map(|normal| normal.sample(rng) as f32)
          .unwrap_or(0.0)
      ```

      Because `read_event` advances virtual time without wall-clock pacing, examples and tests can generate a very long
      sequence quickly, increasing the chance that extreme Gaussian tails enter the diversity-seeking buffer.
    - **Recommended fix:** Add a configurable physical magnetometer range and clamp readings or report saturation when
      the distorted noisy value exceeds it.
