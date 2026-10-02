# AGENTS.md - XREAL Air Packet Log Fixture Guide

## Project Information

Refer to the [driver guide](../../AGENTS.md) for the crate architecture, guardrails, and documentation format, and to
[README.md](../../README.md) for the project overview, supported devices, and protocol blog posts.

## Fixture Format

Each file in this directory is a versioned XREAL Air packet log consumed by `XrealAirReplay::open` in
`__module~/driver/src/xreal_air.rs`. The first line is the header; every remaining line is one exact HID packet in
lowercase hex, two characters per byte, sized by the model's `imu_packet_size`. The replay validates the header
strictly and rejects a wrong magic, a missing model, missing or invalid calibration JSON, and extra header fields.

The header carries three tab-separated fields:

- `# {PACKET_LOG_MAGIC}`: Format magic and version, `ar-drivers-xreal-air-packets-v1`.
- Model name: Written by `AirModel::packet_log_name` and mapped back by `AirModel::from_packet_log_name`; also
  determines the expected packet size.
- Calibration JSON: The glasses' factory IMU calibration for the primary IMU device (`IMU.device_1`), serialized
  verbatim by `ImuDevice::start_packet_logging` and parsed back by `XrealAirBase::from_calibration`.

## Header Calibration Fields

The calibration JSON holds the data the glasses' firmware reports for `IMU.device_1`. The SDK reads the config over
HID (command 0x14 returns the byte length, command 0x15 returns the JSON body in chunks) and performs no
transformation beyond JSON parsing (`get_config_json`). The field names are the firmware's own, and the field
semantics follow the reference implementations of this JSON, notably ar-glass-lib's `XrealFactoryCalibration`
(`taowen/ar-glass-lib`). The SDK consumes only `gyro_bias` and `accel_bias`; every other field passes through unused.
Values shown are the Air 1 capture in `xreal_air_air1_60s.log`, rounded where long.

### Biases and Scales (Per-Sensor Intrinsics)

- **`accel_bias`** `[0.0307, -0.0324, -0.0012]`: Accelerometer hard-iron offset in m/s²; the SDK subtracts it after
  unit conversion in `decode_sensor_report`.
- **`gyro_bias`** `[-0.0106, -0.0071, 0.0053]`: Gyroscope hard-iron offset in rad/s; the SDK subtracts it after the
  raw-to-degrees-to-radians conversion in `decode_sensor_report`.
- **`mag_bias`** `[0, 0, 0]`: Magnetometer hard-iron offset; neutral in this capture.
- **`scale_accel`, `scale_gyro`, `scale_mag`** `[1, 1, 1]`: Per-axis scale gains, the diagonal soft-iron correction
  of each sensor; all unity in this capture.
- **`skew_accel`, `skew_gyro`, `skew_mag`** `[0, 0, 0]`: Axis non-orthogonality terms; reference consumers fold each
  sensor's scale gains and skew terms into one upper-triangular $3 \times 3$ correction matrix whose diagonal holds
  the scale gains and whose superdiagonal holds the skew terms; all zero in this capture.

### Mount Rotation Extrinsics

- **`gyro_q_mag`** `[0.353553, 0.612372, 0.353553, 0.612372]`: Magnetometer-to-gyroscope frame rotation as a
  `[x, y, z, w]` quaternion — a $\arccos(-1/4) \approx 104.48^\circ$ rotation about the axis
  $(1, \sqrt{3}, 1)/\sqrt{5}$, so the magnetometer is mounted at a non-axis-aligned angle relative to the gyroscope.
- **`accel_q_gyro`** `[0, 0, 0, 1]`: Accelerometer-to-gyroscope frame rotation; identity in this capture.
- The firmware stores both quaternions in the passive JPL convention, the transpose of the active Hamilton rotation
  matrix, so consumers invert them (conjugate for unit quaternions) before composing active rotations; together they
  align the magnetometer frame with the accelerometer frame through the gyroscope frame.

### Firmware Fusion Parameters (Passed Through, Unused)

- **`gyro_g_sensitivity`** `[0, 0, 0, 0, 0, 0, 0, 0, 0]`: $3 \times 3$ g-sensitivity matrix, the apparent gyroscope
  rate induced by linear acceleration; all zeros in this capture.
- **`imu_noises`** `[0.00035, 0.00001, 0.00667, 0.00068]`: Four IMU noise standard deviations; the firmware's
  per-slot parameterization is undocumented, most plausibly a gyroscope pair then an accelerometer pair.
- **`gyro_p_mag`** `[0.03096, 0.00535, 0.00382]`: Magnetometer lever arm relative to the gyroscope by the firmware's
  position/quaternion naming, as in the camera descriptors' `imu_p_cam`/`imu_q_cam` pair in the same config; a
  uniform-field lever arm does not affect the magnetic field direction, so consumers ignore it.
