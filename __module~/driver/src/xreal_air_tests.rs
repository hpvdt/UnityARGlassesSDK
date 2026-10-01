use super::*;

const CALIBRATION: &str = concat!(
    r#"{"accel_bias":[0.0,0.0,0.0],"#,
    r#""gyro_bias":[0.0,0.0,0.0],"#,
    r#""gyro_q_mag":[0.0,0.0,0.0,1.0]}"#,
);
const TEST_SENSOR_TIMESTAMP_NANOS: u64 = 0x0102_0304_0506_0708;

fn sensor_packet(timestamp: u64) -> [u8; 0x40] {
    let mut packet = [0; 0x40];
    packet[0] = 1;
    packet[1] = 2;
    packet[4..12].copy_from_slice(&(timestamp * 1000).to_le_bytes());
    packet[12..14].copy_from_slice(&1u16.to_le_bytes());
    packet[14..18].copy_from_slice(&1u32.to_le_bytes());
    packet[27..29].copy_from_slice(&1u16.to_le_bytes());
    packet[29..33].copy_from_slice(&1u32.to_le_bytes());
    packet
}

fn set_version2_magnetometer(
    packet: &mut [u8; 0x40],
    offset: u16,
    divisor: u32,
    samples: [u16; 3],
    freshness: u8,
) {
    packet[42..44].copy_from_slice(&offset.to_le_bytes());
    packet[44..48].copy_from_slice(&divisor.to_le_bytes());
    packet[48..50].copy_from_slice(&samples[0].to_le_bytes());
    packet[50..52].copy_from_slice(&samples[1].to_le_bytes());
    packet[52..54].copy_from_slice(&samples[2].to_le_bytes());
    packet[54..62].copy_from_slice(&TEST_SENSOR_TIMESTAMP_NANOS.to_le_bytes());
    packet[62] = freshness;
}

fn base() -> XrealAirBase {
    XrealAirBase::from_calibration(&CALIBRATION.parse().unwrap()).unwrap()
}

fn encode_packet(packet: &[u8]) -> String {
    packet.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_i24_le(target: &mut [u8], value: i32) {
    target.copy_from_slice(&value.to_le_bytes()[..3]);
}

fn assert_vector_close(actual: Vector3<f32>, expected: Vector3<f32>) {
    assert!(
        (actual - expected).norm() < 1.0e-5,
        "actual={actual:?}, expected={expected:?}"
    );
}

fn packet_log(packets: &[[u8; 0x40]]) -> String {
    let data = packets
        .iter()
        .map(|packet| encode_packet(packet))
        .collect::<Vec<_>>()
        .join("\n");
    format!("# {PACKET_LOG_MAGIC}\tair\t{CALIBRATION}\n{data}\n")
}

fn accgyro_timestamp(event: GlassesEvent) -> u64 {
    match event {
        GlassesEvent::AccGyro { timestamp, .. } => timestamp,
        _ => panic!("expected accelerometer/gyroscope event"),
    }
}

#[test]
fn replay_cycles_through_packets() {
    let mut replay =
        XrealAirReplay::from_packet_log(&packet_log(&[sensor_packet(11), sensor_packet(22)]))
            .unwrap();

    assert_eq!(accgyro_timestamp(replay.read_event().unwrap()), 11);
    assert_eq!(accgyro_timestamp(replay.read_event().unwrap()), 22);
    assert_eq!(accgyro_timestamp(replay.read_event().unwrap()), 11);
}

#[test]
fn base_applies_factory_bias_after_sensor_conversion() {
    let calibration: JsonValue = concat!(
        r#"{"accel_bias":[0.4,0.5,0.6],"#,
        r#""gyro_bias":[0.1,0.2,0.3],"#,
        r#""gyro_q_mag":[0.0,0.0,0.0,1.0]}"#,
    )
    .parse()
    .unwrap();
    let mut base = XrealAirBase::from_calibration(&calibration).unwrap();
    let mut packet = sensor_packet(7);
    packet[12..14].copy_from_slice(&2u16.to_le_bytes());
    packet[14..18].copy_from_slice(&4u32.to_le_bytes());
    write_i24_le(&mut packet[18..21], 120);
    write_i24_le(&mut packet[21..24], 240);
    write_i24_le(&mut packet[24..27], -360);
    packet[27..29].copy_from_slice(&3u16.to_le_bytes());
    packet[29..33].copy_from_slice(&2u32.to_le_bytes());
    write_i24_le(&mut packet[33..36], 2);
    write_i24_le(&mut packet[36..39], -4);
    write_i24_le(&mut packet[39..42], 6);

    base.push_packet(&packet).unwrap();
    let event = base.pop_event().unwrap();
    let GlassesEvent::AccGyro {
        accelerometer,
        gyroscope,
        timestamp,
    } = event
    else {
        panic!("expected accelerometer/gyroscope event");
    };

    assert_eq!(timestamp, 7);
    assert_vector_close(
        gyroscope,
        Vector3::new(
            -60.0f32.to_radians() - 0.1,
            -180.0f32.to_radians() + 0.2,
            120.0f32.to_radians() + 0.3,
        ),
    );
    assert_vector_close(
        accelerometer,
        Vector3::new(-3.0 * 9.8 - 0.4, 9.0 * 9.8 + 0.5, -6.0 * 9.8 + 0.6),
    );
    assert!(base.pop_event().is_none());
}

#[test]
fn gyroscope_scales_signed_boundaries_before_rounding_to_float() {
    let mut base = base();
    let mut packet = sensor_packet(13);
    packet[12..14].copy_from_slice(&u16::MAX.to_le_bytes());
    packet[14..18].copy_from_slice(&16_777_217u32.to_le_bytes());
    write_i24_le(&mut packet[18..21], -8_388_608);
    write_i24_le(&mut packet[21..24], 8_388_607);
    write_i24_le(&mut packet[24..27], 1_234_567);
    base.push_packet(&packet).unwrap();
    let GlassesEvent::AccGyro {
        gyroscope,
        timestamp,
        ..
    } = base.pop_event().unwrap()
    else {
        panic!("expected accelerometer/gyroscope event");
    };
    assert_eq!(timestamp, 13);
    // Golden float bits from double-precision scaling and radians conversion.
    assert_eq!(
        gyroscope.map(f32::to_bits),
        Vector3::new(0x440e_f9a6, 0x42a8_55dc, 0x440e_f9a4)
    );
}

#[test]
fn accelerometer_scales_signed_boundaries_before_rounding_to_float() {
    let mut base = base();
    let mut packet = sensor_packet(17);
    packet[27..29].copy_from_slice(&u16::MAX.to_le_bytes());
    packet[29..33].copy_from_slice(&16_777_217u32.to_le_bytes());
    write_i24_le(&mut packet[33..36], -8_388_608);
    write_i24_le(&mut packet[36..39], 8_388_607);
    write_i24_le(&mut packet[39..42], 1_234_567);
    base.push_packet(&packet).unwrap();
    let GlassesEvent::AccGyro {
        accelerometer,
        timestamp,
        ..
    } = base.pop_event().unwrap()
    else {
        panic!("expected accelerometer/gyroscope event");
    };
    assert_eq!(timestamp, 17);
    // Golden float bits from double-precision scaling with the device's gravity factor.
    assert_eq!(
        accelerometer.map(f32::to_bits),
        Vector3::new(0x489c_cc2f, 0x4738_9c0b, 0x489c_cc2e)
    );
}

#[test]
fn version2_magnetometer_is_little_endian_and_mapped_directly_to_rub() {
    let mut base = base();
    let mut packet = sensor_packet(19);
    let offset = 0x1234;
    set_version2_magnetometer(
        &mut packet,
        offset,
        0x0100,
        [offset + 0x0100, offset + 0x0200, offset + 0x0300],
        0xff,
    );

    let decoded = decode_xreal_magnetometer_report(&packet).unwrap();
    assert_eq!(decoded.magnetic_field, Vector3::new(200.0, 300.0, 100.0));
    assert_eq!(decoded.sensor_timestamp_nanos, TEST_SENSOR_TIMESTAMP_NANOS);
    assert!(decoded.fresh);

    base.push_packet(&packet).unwrap();
    let GlassesEvent::Magnetometer {
        magnetometer,
        timestamp,
    } = base.pop_event().unwrap()
    else {
        panic!("magnetometer must precede accelerometer/gyroscope");
    };
    assert_eq!(timestamp, 19);
    assert_eq!(magnetometer, Vector3::new(200.0, 300.0, 100.0));
    assert!(matches!(
        base.pop_event(),
        Some(GlassesEvent::AccGyro { timestamp: 19, .. })
    ));
    assert!(base.pop_event().is_none());
}

#[test]
fn version2_magnetometer_samples_below_offset_remain_negative() {
    let mut base = base();
    let mut packet = sensor_packet(21);
    set_version2_magnetometer(&mut packet, 0x8000, 0x0100, [0x7f00, 0x8100, 0x7e00], 1);

    let decoded = decode_xreal_magnetometer_report(&packet).unwrap();
    assert_eq!(decoded.magnetic_field, Vector3::new(100.0, -200.0, -100.0));
    base.push_packet(&packet).unwrap();
    let GlassesEvent::Magnetometer { magnetometer, .. } = base.pop_event().unwrap() else {
        panic!("expected magnetic event");
    };
    assert_eq!(magnetometer, decoded.magnetic_field);
}

#[test]
fn upstream_decoder_rejects_invalid_report_envelopes() {
    assert!(decode_xreal_magnetometer_report(&[1, 2]).is_none());
    let mut packet = sensor_packet(1);
    packet[0] = 0;
    assert!(decode_xreal_magnetometer_report(&packet).is_none());
    packet[0] = 1;
    packet[1] = 3;
    assert!(decode_xreal_magnetometer_report(&packet).is_none());
}

#[test]
fn truncated_sensor_reports_return_errors_without_queuing_events() {
    let mut base = base();
    let packet = sensor_packet(37);
    for length in 0..64 {
        assert!(matches!(
            base.decode_sensor_report(&packet[..length]),
            Err(Error::Other("XREAL sensor report is shorter than 64 bytes"))
        ));
        if length >= 2 {
            assert!(base.push_packet(&packet[..length]).is_err());
        } else {
            base.push_packet(&packet[..length]).unwrap();
        }
        assert!(base.pop_event().is_none());
    }
}

#[test]
fn padded_sensor_reports_decode_the_complete_sensor_prefix() {
    let mut base = base();
    let mut packet = sensor_packet(41);
    set_version2_magnetometer(&mut packet, 100, 10, [110, 120, 130], 1);
    for length in [64, 128, 512] {
        let mut padded = vec![0xa5; length];
        padded[..64].copy_from_slice(&packet);
        base.push_packet(&padded).unwrap();
        let GlassesEvent::Magnetometer {
            magnetometer,
            timestamp,
        } = base.pop_event().unwrap()
        else {
            panic!("expected magnetic event before accelerometer/gyroscope");
        };
        assert_eq!(timestamp, 41);
        assert_eq!(magnetometer, Vector3::new(200.0, 300.0, 100.0));
        assert!(matches!(
            base.pop_event(),
            Some(GlassesEvent::AccGyro { timestamp: 41, .. })
        ));
        assert!(base.pop_event().is_none());
    }
}

#[test]
fn captured_packet_matches_upstream_deserialization() {
    let trace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xreal_air_air1_60s.log");
    let packet_log = std::fs::read_to_string(trace).unwrap();
    let packet = decode_packet_log_line(packet_log.lines().nth(6).unwrap(), 0x40).unwrap();

    let decoded = decode_xreal_magnetometer_report(&packet).unwrap();
    assert_eq!(
        decoded.magnetic_field,
        Vector3::new(62_100.0 / 1024.0, 93_200.0 / 1024.0, 1_600.0 / 1024.0)
    );
    assert_eq!(decoded.sensor_timestamp_nanos, 22_961_000);
    assert!(decoded.fresh);
}

#[test]
fn cached_magnetometer_is_not_emitted() {
    let mut base = base();
    let mut packet = sensor_packet(23);
    set_version2_magnetometer(&mut packet, 100, 10, [110, 120, 130], 0);

    let decoded = decode_xreal_magnetometer_report(&packet).unwrap();
    assert_eq!(decoded.magnetic_field, Vector3::new(200.0, 300.0, 100.0));
    assert!(!decoded.fresh);

    base.push_packet(&packet).unwrap();
    assert!(matches!(
        base.pop_event(),
        Some(GlassesEvent::AccGyro { timestamp: 23, .. })
    ));
    assert!(base.pop_event().is_none());
}

#[test]
fn invalid_magnetometer_does_not_drop_accgyro() {
    let mut base = base();
    let mut zero_divisor = sensor_packet(29);
    set_version2_magnetometer(&mut zero_divisor, 100, 0, [110, 120, 130], 1);
    let mut zero_norm = sensor_packet(31);
    set_version2_magnetometer(&mut zero_norm, 100, 10, [100, 100, 100], 1);

    let decoded = decode_xreal_magnetometer_report(&zero_divisor).unwrap();
    assert!(decoded
        .magnetic_field
        .iter()
        .any(|component| !component.is_finite()));

    base.push_packet(&zero_divisor).unwrap();
    assert!(matches!(
        base.pop_event(),
        Some(GlassesEvent::AccGyro { timestamp: 29, .. })
    ));
    base.push_packet(&zero_norm).unwrap();
    assert!(matches!(
        base.pop_event(),
        Some(GlassesEvent::AccGyro { timestamp: 31, .. })
    ));
    assert!(base.pop_event().is_none());
}

#[test]
fn non_finite_magnetometer_is_invalid() {
    assert!(!is_valid_magnetic_observation(&Vector3::new(
        f32::NAN,
        1.0,
        1.0
    )));
    assert!(!is_valid_magnetic_observation(&Vector3::new(
        f32::INFINITY,
        1.0,
        1.0
    )));
}

#[test]
fn replay_rejects_malformed_logs() {
    let packet = encode_packet(&sensor_packet(1));
    let header = format!("# {PACKET_LOG_MAGIC}\tair\t{CALIBRATION}");
    let malformed = [
        String::new(),
        format!("# wrong\tair\t{CALIBRATION}\n{packet}\n"),
        format!("# {PACKET_LOG_MAGIC}\tunknown\t{CALIBRATION}\n{packet}\n"),
        format!("# {PACKET_LOG_MAGIC}\tair\tnot-json\n{packet}\n"),
        format!("{header}\textra\n{packet}\n"),
        format!("# {PACKET_LOG_MAGIC}\tair\t{{}}\n{packet}\n"),
        format!("{header}\n"),
        format!("{header}\n{}\n", &packet[..packet.len() - 1]),
        format!("{header}\n{}A\n", &packet[..packet.len() - 1]),
        format!("{header}\n{packet}\n\n{packet}\n"),
    ];

    for packet_log in malformed {
        assert!(XrealAirReplay::from_packet_log(&packet_log).is_err());
    }
}

#[test]
fn replay_rejects_packet_length_for_model() {
    let packet = encode_packet(&sensor_packet(1));
    let packet_log = format!("# {PACKET_LOG_MAGIC}\tair2-ultra\t{CALIBRATION}\n{packet}\n");
    assert!(XrealAirReplay::from_packet_log(&packet_log).is_err());
}

#[test]
fn replay_hardware_operations_are_unsupported() {
    let mut replay = XrealAirReplay::from_packet_log(&packet_log(&[sensor_packet(1)])).unwrap();

    assert!(matches!(replay.serial(), Err(Error::NotImplemented)));
    assert!(matches!(
        replay.get_display_mode(),
        Err(Error::NotImplemented)
    ));
    assert!(matches!(
        replay.set_display_mode(DisplayMode::SameOnBoth),
        Err(Error::NotImplemented)
    ));
    assert!(matches!(
        replay.display_matrices(),
        Err(Error::NotImplemented)
    ));
}

fn magnetic_factory_calibration() -> JsonValue {
    let mut calibration: JsonValue = CALIBRATION.parse().unwrap();
    let object = calibration.get_mut::<HashMap<String, JsonValue>>().unwrap();
    object.insert("mag_bias".into(), "[0,0,0]".parse().unwrap());
    object.insert("scale_mag".into(), "[1,1,1]".parse().unwrap());
    calibration
}

fn set_calibration_field(calibration: &mut JsonValue, name: &str, value: &str) {
    calibration
        .get_mut::<HashMap<String, JsonValue>>()
        .unwrap()
        .insert(name.into(), value.parse().unwrap());
}

fn decoded_magnetic_field(calibration: &JsonValue, samples: [u16; 3]) -> Vector3<f32> {
    let mut base = XrealAirBase::from_calibration(calibration).unwrap();
    let mut packet = sensor_packet(43);
    set_version2_magnetometer(&mut packet, 100, 100, samples, 1);
    base.push_packet(&packet).unwrap();
    let GlassesEvent::Magnetometer { magnetometer, .. } = base.pop_event().unwrap() else {
        panic!("expected factory-calibrated magnetic event");
    };
    assert!(matches!(
        base.pop_event(),
        Some(GlassesEvent::AccGyro { timestamp: 43, .. })
    ));
    assert!(base.pop_event().is_none());
    magnetometer
}

#[test]
fn factory_magnetic_alignment_uses_passive_quaternion_in_native_axes() {
    let mut calibration = magnetic_factory_calibration();
    set_calibration_field(
        &mut calibration,
        "gyro_q_mag",
        "[0.353553,0.612372,0.353553,0.612372]",
    );
    // Independent native-axis golden vectors for the captured factory rotation.
    // The six-decimal quaternion limits coefficient accuracy to about 1e-6;
    // allow 1e-4 at these 100-microtesla test magnitudes.
    for (samples, expected) in [
        ([200, 100, 100], Vector3::new(0.0, 0.0, -100.0)),
        ([100, 200, 100], Vector3::new(86.60252, -50.00003, 0.0)),
        ([100, 100, 200], Vector3::new(-50.00003, -86.60252, 0.0)),
    ] {
        let actual = decoded_magnetic_field(&calibration, samples);
        assert!(
            (actual - expected).norm() < 1e-4,
            "actual={actual:?}, expected={expected:?}"
        );
    }
    let reference = decoded_magnetic_field(&calibration, [200, 300, 400]);
    set_calibration_field(
        &mut calibration,
        "gyro_q_mag",
        "[0.707106,1.224744,0.707106,1.224744]",
    );
    assert_vector_close(
        decoded_magnetic_field(&calibration, [200, 300, 400]),
        reference,
    );
}

#[test]
fn factory_identity_alignment_converts_native_magnetic_axes_to_rub() {
    assert_vector_close(
        decoded_magnetic_field(&magnetic_factory_calibration(), [120, 130, 140]),
        Vector3::new(20.0, -30.0, -40.0),
    );
}

#[test]
fn factory_magnetic_alignment_composes_accelerometer_extrinsics() {
    let mut calibration = magnetic_factory_calibration();
    set_calibration_field(&mut calibration, "gyro_q_mag", "[1,0,0,1]");
    set_calibration_field(&mut calibration, "accel_q_gyro", "[0,0,1,1]");
    // The passive magnetic rotation is -90 degrees about X, followed by the
    // inverse active acceleration-to-gyro rotation, -90 degrees about Z.
    // These do not commute; the final RUB transform negates Y and Z.
    assert_vector_close(
        decoded_magnetic_field(&calibration, [120, 130, 140]),
        Vector3::new(40.0, 20.0, 30.0),
    );
}

#[test]
fn factory_magnetic_alignment_preserves_accelerometer_and_gyroscope() {
    let mut calibration = magnetic_factory_calibration();
    set_calibration_field(&mut calibration, "gyro_q_mag", "[1,2,3,4]");
    set_calibration_field(&mut calibration, "accel_q_gyro", "[4,3,2,1]");
    let mut calibrated = XrealAirBase::from_calibration(&calibration).unwrap();
    let mut legacy = base();
    let mut packet = sensor_packet(53);
    write_i24_le(&mut packet[18..21], 17);
    write_i24_le(&mut packet[21..24], -31);
    write_i24_le(&mut packet[24..27], 43);
    write_i24_le(&mut packet[33..36], -7);
    write_i24_le(&mut packet[36..39], 11);
    write_i24_le(&mut packet[39..42], -19);
    set_version2_magnetometer(&mut packet, 100, 100, [120, 130, 140], 1);
    calibrated.push_packet(&packet).unwrap();
    legacy.push_packet(&packet).unwrap();
    calibrated.pop_event().unwrap();
    legacy.pop_event().unwrap();
    let GlassesEvent::AccGyro {
        accelerometer: expected_acc,
        gyroscope: expected_gyro,
        timestamp: expected_time,
    } = legacy.pop_event().unwrap()
    else {
        panic!("expected inertial event");
    };
    let GlassesEvent::AccGyro {
        accelerometer,
        gyroscope,
        timestamp,
    } = calibrated.pop_event().unwrap()
    else {
        panic!("expected inertial event");
    };
    assert_eq!(accelerometer, expected_acc);
    assert_eq!(gyroscope, expected_gyro);
    assert_eq!(timestamp, expected_time);
}
