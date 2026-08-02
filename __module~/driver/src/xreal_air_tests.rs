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
fn base_preserves_accelerometer_and_gyroscope_decoding() {
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
        Vector3::new(-3.0 * 9.81 - 0.4, 9.0 * 9.81 + 0.5, -6.0 * 9.81 + 0.6),
    );
    assert!(base.pop_event().is_none());
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
fn upstream_version1_magnetometer_layout_is_deserialized_by_absolute_offset() {
    let mut packet = [0u8; 0x40];
    packet[0] = 1;
    packet[1] = 1;
    let offset = 0x1234u16;
    packet[36..38].copy_from_slice(&offset.to_le_bytes());
    packet[38..42].copy_from_slice(&0x0100u32.to_le_bytes());
    packet[42..44].copy_from_slice(&(offset + 0x0100).to_le_bytes());
    packet[44..46].copy_from_slice(&(offset + 0x0200).to_le_bytes());
    packet[46..48].copy_from_slice(&(offset + 0x0300).to_le_bytes());
    packet[48..56].copy_from_slice(&TEST_SENSOR_TIMESTAMP_NANOS.to_le_bytes());
    packet[56] = 1;

    let decoded = decode_xreal_magnetometer_report(&packet).unwrap();
    assert_eq!(decoded.magnetic_field, Vector3::new(200.0, 300.0, 100.0));
    assert_eq!(decoded.sensor_timestamp_nanos, TEST_SENSOR_TIMESTAMP_NANOS);
    assert!(decoded.fresh);
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
