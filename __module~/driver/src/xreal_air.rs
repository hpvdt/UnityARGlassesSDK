// Copyright (C) 2023, Alex Badics
// This file is part of ar-drivers-rs
// Licensed under the MIT license. See LICENSE file in the project root for details.

// Based on code by thejackimonster
// See https://gitlab.com/TheJackiMonster/nrealAirLinuxDriver

//! XREAL Air AR glasses support. See [`XrealAir`]
//! It only uses [`hidapi`] for communication.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use hidapi::{HidApi, HidDevice};
use nalgebra::{Isometry3, Matrix3, Quaternion, Translation3, UnitQuaternion, Vector3};
use tinyjson::JsonValue;

use crate::{
    util::crc32_adler, ARGlasses, DisplayMatrices, DisplayMode, Error, GlassesEvent, Result, Side,
};

/// The main structure representing a connected XREAL Air glasses
pub struct XrealAir {
    model: AirModel,
    device: HidDevice,
    pending_packets: VecDeque<McuPacket>,
    imu_device: ImuDevice,
}

const COMMAND_TIMEOUT: i32 = 1000;
const IMU_TIMEOUT: i32 = 250;
const PACKET_LOG_MAGIC: &str = "ar-drivers-xreal-air-packets-v1";

const XREAL_VID: u16 = 0x3318;
const AIR_PID: u16 = 0x0424;
const AIR_2_PID: u16 = 0x0428;
const AIR_2_PRO_PID: u16 = 0x0432;
const AIR_2_ULTRA_PID: u16 = 0x0426;

/// Describes the particular Air model.
#[derive(Debug, Clone, Copy)]
pub enum AirModel {
    /// XREAL Air (original)
    Air,
    /// XREAL Air 2
    Air2,
    /// XREAL Air 2 Pro
    Air2Pro,
    /// XREAL Air 2 Ultra
    Air2Ultra,
}

impl TryFrom<u16> for AirModel {
    type Error = Error;

    fn try_from(val: u16) -> Result<AirModel> {
        match val {
            AIR_PID => Ok(AirModel::Air),
            AIR_2_PID => Ok(AirModel::Air2),
            AIR_2_PRO_PID => Ok(AirModel::Air2Pro),
            AIR_2_ULTRA_PID => Ok(AirModel::Air2Ultra),
            _ => Err(Error::Other("unsupported XREAL product")),
        }
    }
}

impl AirModel {
    fn from_packet_log_name(name: &str) -> Result<Self> {
        match name {
            "air" => Ok(AirModel::Air),
            "air2" => Ok(AirModel::Air2),
            "air2-pro" => Ok(AirModel::Air2Pro),
            "air2-ultra" => Ok(AirModel::Air2Ultra),
            _ => Err(Error::Other("Unknown model in packet log")),
        }
    }

    fn packet_log_name(&self) -> &'static str {
        match self {
            AirModel::Air => "air",
            AirModel::Air2 => "air2",
            AirModel::Air2Pro => "air2-pro",
            AirModel::Air2Ultra => "air2-ultra",
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            AirModel::Air => "XREAL Air",
            AirModel::Air2 => "XREAL Air 2",
            AirModel::Air2Pro => "XREAL Air 2 Pro",
            AirModel::Air2Ultra => "XREAL Air 2 Ultra",
        }
    }

    /// Returns the MCU/command interface number for this model
    fn mcu_interface(&self) -> i32 {
        match self {
            AirModel::Air2Ultra => 0,
            _ => 4,
        }
    }

    /// Returns the IMU interface number for this model
    fn imu_interface(&self) -> i32 {
        match self {
            AirModel::Air2Ultra => 2,
            _ => 3,
        }
    }

    /// Returns the maximum IMU packet size for this model
    fn imu_packet_size(&self) -> usize {
        match self {
            AirModel::Air2Ultra => 0x200,
            _ => 0x40,
        }
    }
}

impl ARGlasses for XrealAir {
    fn serial(&mut self) -> Result<String> {
        let mut result = self.run_command(McuPacket {
            cmd_id: 0x15,
            ..Default::default()
        })?;
        result.remove(0);
        String::from_utf8(result).map_err(|_| Error::Other("Serial number was not utf-8"))
    }

    fn read_event(&mut self) -> Result<GlassesEvent> {
        if let Some(event) = self.read_mcu_packet()? {
            Ok(event)
        } else {
            self.imu_device.read_packet()
        }
    }

    fn start_packet_logging(&mut self, path: &Path) -> Result<()> {
        self.imu_device.start_packet_logging(path)
    }

    fn stop_packet_logging(&mut self) -> Result<()> {
        self.imu_device.stop_packet_logging()
    }

    fn get_display_mode(&mut self) -> Result<DisplayMode> {
        let result = self.run_command(McuPacket {
            cmd_id: 0x7,
            ..Default::default()
        })?;
        match result.get(1) {
            // Mirror 60Hz
            Some(1) => Ok(DisplayMode::SameOnBoth),
            // SBS 60Hz
            Some(3) => Ok(DisplayMode::Stereo),
            // SBS 72Hz
            Some(4) => Ok(DisplayMode::HighRefreshRate),
            // Mirror 72Hz
            Some(5) => Ok(DisplayMode::SameOnBoth),
            // Horizontally stretched SBS, 60Hz
            Some(8) => Ok(DisplayMode::HalfSBS),
            // SBS 90Hz
            Some(9) => Ok(DisplayMode::HighRefreshRate),
            // Mirror 90Hz
            Some(10) => Ok(DisplayMode::HighRefreshRate),
            // Mirror 120Hz
            Some(11) => Ok(DisplayMode::HighRefreshRate),
            _ => Err(Error::Other("Unknown display mode")),
        }
    }

    fn set_display_mode(&mut self, display_mode: DisplayMode) -> Result<()> {
        let display_mode_byte = match display_mode {
            DisplayMode::SameOnBoth => 1,
            DisplayMode::HalfSBS => 8,
            DisplayMode::Stereo => 3,
            DisplayMode::HighRefreshRate => 11,
            DisplayMode::HighRefreshRateSBS => 9,
        };
        let result = self.run_command(McuPacket {
            cmd_id: 0x08,
            data: vec![display_mode_byte],
        })?;

        if result.first() == Some(&0) {
            Ok(())
        } else {
            Err(Error::Other("Display mode setting unsuccessful"))
        }
    }

    // TODO
    fn display_fov(&self) -> f32 {
        // This is a judgement call. The displays have a non-trivial distortion,
        // so this value is a bit much in hte middle, a bit too low on the borders.
        24.0f32.to_radians()
    }

    fn imu_to_display_matrix(&self, side: Side, ipd: f32) -> Isometry3<f64> {
        let side_multiplier = match side {
            Side::Left => -0.5,
            Side::Right => 0.5,
        };
        Translation3::new(ipd as f64 * side_multiplier, 0.0, 0.0)
            * UnitQuaternion::from_euler_angles(
                // Apparently there is no noticable tilt
                0.0,
                Self::DISPLAY_DIVERGENCE * side_multiplier,
                0.0,
            )
    }

    fn display_matrices(&self) -> Result<(DisplayMatrices, DisplayMatrices)> {
        self.imu_device.displays.clone().ok_or(Error::NotFound)
    }

    fn display_delay(&self) -> u64 {
        7000
    }

    fn name(&self) -> &'static str {
        self.model.display_name()
    }
}

/// A deterministic replay of raw packets captured from XREAL Air glasses.
pub struct XrealAirReplay {
    model: AirModel,
    packets: Vec<Vec<u8>>,
    next_packet: usize,
    base: XrealAirBase,
}

impl XrealAirReplay {
    /// Open and validate a versioned XREAL Air packet log.
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_packet_log(&fs::read_to_string(path)?)
    }

    fn from_packet_log(packet_log: &str) -> Result<Self> {
        let mut lines = packet_log.lines();
        let header = lines
            .next()
            .ok_or(Error::Other("Packet log is missing its header"))?;
        let mut header_parts = header.split('\t');
        if header_parts.next() != Some(&format!("# {PACKET_LOG_MAGIC}")) {
            return Err(Error::Other("Unsupported packet log header"));
        }
        let model = AirModel::from_packet_log_name(
            header_parts
                .next()
                .ok_or(Error::Other("Packet log is missing its model"))?,
        )?;
        let calibration: JsonValue = header_parts
            .next()
            .ok_or(Error::Other("Packet log is missing IMU calibration"))?
            .parse()
            .map_err(|_| Error::Other("Packet log has invalid IMU calibration JSON"))?;
        if header_parts.next().is_some() {
            return Err(Error::Other("Packet log header has extra fields"));
        }

        let base = XrealAirBase::from_calibration(&calibration)?;
        let expected_packet_size = model.imu_packet_size();
        let packets = lines
            .map(|line| decode_packet_log_line(line, expected_packet_size))
            .collect::<Result<Vec<_>>>()?;
        if packets.is_empty() {
            return Err(Error::Other("Packet log contains no packets"));
        }

        Ok(Self {
            model,
            packets,
            next_packet: 0,
            base,
        })
    }
}

impl ARGlasses for XrealAirReplay {
    fn serial(&mut self) -> Result<String> {
        Err(Error::NotImplemented)
    }

    fn read_event(&mut self) -> Result<GlassesEvent> {
        if let Some(event) = self.base.pop_event() {
            return Ok(event);
        }

        for _ in 0..self.packets.len() {
            let packet = &self.packets[self.next_packet];
            self.next_packet = (self.next_packet + 1) % self.packets.len();
            self.base.push_packet(packet)?;
            if let Some(event) = self.base.pop_event() {
                return Ok(event);
            }
        }

        Err(Error::Other("Packet log contains no sensor reports"))
    }

    fn get_display_mode(&mut self) -> Result<DisplayMode> {
        Err(Error::NotImplemented)
    }

    fn set_display_mode(&mut self, _display_mode: DisplayMode) -> Result<()> {
        Err(Error::NotImplemented)
    }

    fn display_fov(&self) -> f32 {
        24.0f32.to_radians()
    }

    fn imu_to_display_matrix(&self, side: Side, ipd: f32) -> Isometry3<f64> {
        let side_multiplier = match side {
            Side::Left => -0.5,
            Side::Right => 0.5,
        };
        Translation3::new(ipd as f64 * side_multiplier, 0.0, 0.0)
            * UnitQuaternion::from_euler_angles(
                0.0,
                XrealAir::DISPLAY_DIVERGENCE * side_multiplier,
                0.0,
            )
    }

    fn name(&self) -> &'static str {
        self.model.display_name()
    }

    fn display_matrices(&self) -> Result<(DisplayMatrices, DisplayMatrices)> {
        Err(Error::NotImplemented)
    }

    fn display_delay(&self) -> u64 {
        7000
    }
}

fn parse_calibration_vector(calibration: &JsonValue, name: &str) -> Result<Vector3<f32>> {
    let object = calibration
        .get::<HashMap<String, JsonValue>>()
        .ok_or(Error::Other("IMU calibration must be a JSON object"))?;
    let values = object
        .get(name)
        .and_then(|value| value.get::<Vec<JsonValue>>())
        .ok_or(Error::Other("IMU calibration vector is missing or invalid"))?;
    if values.len() != 3 {
        return Err(Error::Other("IMU calibration vector has invalid length"));
    }
    let mut components = [0.0; 3];
    for (component, value) in components.iter_mut().zip(values) {
        *component = *value
            .get::<f64>()
            .ok_or(Error::Other("IMU calibration vector contains a non-number"))?
            as f32;
    }
    Ok(Vector3::from(components))
}

fn decode_packet_log_line(line: &str, expected_packet_size: usize) -> Result<Vec<u8>> {
    if line.len() != expected_packet_size * 2 {
        return Err(Error::Other("Packet log line has invalid length"));
    }
    if !line
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::Other(
            "Packet log data must be lowercase hexadecimal",
        ));
    }
    (0..expected_packet_size)
        .map(|index| {
            u8::from_str_radix(&line[index * 2..index * 2 + 2], 16)
                .map_err(|_| Error::Other("Packet log contains invalid hexadecimal"))
        })
        .collect()
}

fn is_valid_magnetic_observation(magnetometer: &Vector3<f32>) -> bool {
    let norm_squared = magnetometer.norm_squared();
    magnetometer.iter().all(|component| component.is_finite())
        && norm_squared.is_finite()
        && norm_squared > 0.0
}

#[derive(Debug, Clone, Copy)]
struct XrealMagnetometerReport {
    magnetic_field: Vector3<f32>,
    sensor_timestamp_nanos: u64,
    fresh: bool,
}

// Ported from ar-glass-lib's decode_xreal_imu at
// c172403f8df2108de5708c8663bb4c2359b0bf5b.
fn decode_xreal_magnetometer_report(report: &[u8]) -> Option<XrealMagnetometerReport> {
    if report.len() < 64 || report[0] != 1 {
        return None;
    }
    let (offset_field, denominator_field, values_field, sensor_timestamp_field, freshness_field) =
        match report[1] {
            1 => (36, 38, 42, 48, 56),
            2 => (42, 44, 48, 54, 62),
            _ => return None,
        };
    let offset = LittleEndian::read_u16(&report[offset_field..]) as f64;
    let denominator = LittleEndian::read_u32(&report[denominator_field..]) as f64;
    let mut scaled = [0.0f32; 3];
    for (index, value) in scaled.iter_mut().enumerate() {
        let raw = LittleEndian::read_u16(&report[values_field + index * 2..]) as f64;
        *value = (100.0 * (raw - offset) / denominator) as f32;
    }
    Some(XrealMagnetometerReport {
        // The ar-glass-lib transcription kept positive signs on axes 1 and 2, which
        // breaks the constant magnetic-gravity inner product on the captured
        // calibration dance. The fixture requires (scaled[1], -scaled[2], -scaled[0]).
        magnetic_field: Vector3::new(scaled[1], -scaled[2], -scaled[0]),
        sensor_timestamp_nanos: LittleEndian::read_u64(&report[sensor_timestamp_field..]),
        fresh: report[freshness_field] != 0,
    })
}

impl XrealAir {
    /// Vendor ID of the XREAL Air's components
    pub const VID: u16 = XREAL_VID;
    /// Product ID of the XREAL Air 1's components
    #[deprecated]
    pub const PID: u16 = AIR_PID;

    const DISPLAY_DIVERGENCE: f64 = 0.017;

    /// Connect to a specific glasses, based on the
    /// Mainly made to work around android permission issues
    #[cfg(target_os = "android")]
    pub fn new(fd: isize) -> Result<Self> {
        // First, get the device info to determine the model
        let hidapi = HidApi::new_without_enumerate()?;
        // Try MCU interface 4 first (for Air, Air 2, Air 2 Pro), then 0 (for Air 2 Ultra)
        let (device, model) = if let Ok(dev) = hidapi.wrap_sys_device(fd, 4) {
            let pid = dev.get_device_info()?.product_id();
            (dev, AirModel::try_from(pid)?)
        } else {
            let dev = hidapi.wrap_sys_device(fd, 0)?;
            let pid = dev.get_device_info()?.product_id();
            (dev, AirModel::try_from(pid)?)
        };
        Self::new_common(model, device, ImuDevice::new(fd, model)?)
    }

    /// Find a connected XREAL Air device and connect to it. (And claim the USB interface)
    /// Only one instance can be alive at a time
    #[cfg(not(target_os = "android"))]
    pub fn new() -> Result<Self> {
        let (model, mcu_device, imu_device) = open_xreal_air()?;
        Self::new_common(model, mcu_device, ImuDevice::new_device(imu_device, model)?)
    }
    fn new_common(model: AirModel, device: HidDevice, imu_device: ImuDevice) -> Result<Self> {
        let mut result = Self {
            model,
            device,
            pending_packets: Default::default(),
            imu_device,
        };
        // Quick check
        result.serial()?;
        Ok(result)
    }

    /// Returns the calibration data stored on the Glasses. No transformation
    /// is done on the data, except for Json Parsing.
    pub fn get_config_json(&self) -> &JsonValue {
        &self.imu_device.config_json
    }

    fn read_mcu_packet(&mut self) -> Result<Option<GlassesEvent>> {
        let packet = if let Some(packet) = self.pending_packets.pop_front() {
            packet
        } else if let Some(packet) = self.read_packet(0)? {
            packet
        } else {
            return Ok(None);
        };
        Ok(match packet {
            McuPacket {
                cmd_id: 0x6c05,
                data,
            } => Some(GlassesEvent::KeyPress(data[0] - 1)),
            // NOTE: maybe we should retry in these cases instead of basically reporting timeout,
            //       but we will be called again soon enough.
            McuPacket {
                cmd_id: 0x6c09,
                data: _data,
            } => {
                // TODO: optional logging in the crate
                // eprintln!("Got error: {}", String::from_utf8(_data).unwrap());
                None
            }
            _ => None,
        })
    }

    fn read_packet(&mut self, timeout: i32) -> Result<Option<McuPacket>> {
        let mut result = [0u8; 0x40];
        let packet_size = self.device.read_timeout(&mut result, timeout)?;
        if packet_size == 0 {
            Ok(None)
        } else {
            Ok(Some(
                McuPacket::deserialize(&result).ok_or(Error::Other("Malformed packet received"))?,
            ))
        }
    }

    fn run_command(&mut self, command: McuPacket) -> Result<Vec<u8>> {
        let expected_cmd_id = command.cmd_id;
        let packet = command
            .serialize()
            .ok_or(Error::Other("Packet serialization failed"))?;
        write_hid_packet(&self.device, &packet)?;

        for _ in 0..64 {
            let packet = self
                .read_packet(COMMAND_TIMEOUT)?
                .ok_or(Error::PacketTimeout)?;
            if packet.cmd_id == expected_cmd_id {
                return Ok(packet.data);
            }
            self.pending_packets.push_back(packet);
        }
        Err(Error::Other("Received too many unrelated packets"))
    }
}

struct ImuDevice {
    device: HidDevice,
    model: AirModel,
    config_json: JsonValue,
    displays: Option<(DisplayMatrices, DisplayMatrices)>,
    base: XrealAirBase,
    packet_log: Option<BufWriter<File>>,
}

impl ImuDevice {
    #[cfg(target_os = "android")]
    pub fn new(fd: isize, model: AirModel) -> Result<Self> {
        let imu_iface = model.imu_interface();
        Self::new_device(
            HidApi::new_without_enumerate()?.wrap_sys_device(fd, imu_iface)?,
            model,
        )
    }

    fn new_device(device: HidDevice, model: AirModel) -> Result<Self> {
        let mut result = Self {
            device,
            model,
            config_json: JsonValue::Null,
            displays: None,
            base: XrealAirBase::default(),
            packet_log: None,
        };
        // Turn off IMU stream while reading config
        result.command(0x19, &[0x0])?;
        result.read_config()?;
        result.parse_config()?;
        // Turn IMU stream back on
        result.command(0x19, &[0x1])?;

        Ok(result)
    }

    fn read_config(&mut self) -> Result<()> {
        let len = u32::from_le_bytes(self.command(0x14, &[])?.try_into().unwrap());
        let mut config = Vec::new();
        while config.len() < len as usize {
            let mut config_part = self.command(0x15, &[])?;
            config.append(&mut config_part);
        }
        let config_as_str = String::from_utf8(config)
            .map_err(|_| Error::Other("Invalid glasses config (not utf-8)"))?;
        self.config_json = config_as_str
            .parse()
            .map_err(|_| Error::Other("Invalid glasses config format (JSON parse error)"))?;
        Ok(())
    }

    fn parse_config(&mut self) -> Result<()> {
        // XXX: This will panic if config is not in expected format.
        //      should probably return Err() instead.
        self.displays = Self::parse_display_descriptors(&self.config_json["display"]);
        let cfg = &self.config_json["IMU"]["device_1"];
        self.base = XrealAirBase::from_calibration(cfg)?;
        Ok(())
    }

    fn parse_display_descriptors(json: &JsonValue) -> Option<(DisplayMatrices, DisplayMatrices)> {
        let resolution = &json["resolution"];
        let resolution = (
            *resolution[0].get::<f64>().unwrap() as u32,
            *resolution[1].get::<f64>().unwrap() as u32,
        );

        let side_descriptor =
            |p_cfg: &JsonValue, q_cfg: &JsonValue, k_cfg: &JsonValue| -> DisplayMatrices {
                let translation = Self::parse_vector(p_cfg);
                let rotation = UnitQuaternion::from_quaternion(Self::parse_quaternion(q_cfg));
                DisplayMatrices {
                    intrinsic_matrix: Self::parse_matrix3(k_cfg),
                    resolution,
                    isometry: Translation3::from(translation) * rotation,
                }
            };
        let mut left = side_descriptor(
            &json["target_p_left_display"],
            &json["target_q_left_display"],
            &json["k_left_display"],
        );
        let mut right = side_descriptor(
            &json["target_p_right_display"],
            &json["target_q_right_display"],
            &json["k_right_display"],
        );
        // The calibration seems to be based on a reference point near the right lens.
        // We will center the translation component between the displays.
        let mean = (left.isometry.translation.vector + right.isometry.translation.vector) * 0.5;
        left.isometry.translation.vector -= mean;
        right.isometry.translation.vector -= mean;
        Some((left, right))
    }

    fn parse_vector(json: &JsonValue) -> Vector3<f64> {
        Vector3::new(
            *json[0].get::<f64>().unwrap(),
            *json[1].get::<f64>().unwrap(),
            *json[2].get::<f64>().unwrap(),
        )
    }

    fn parse_quaternion(json: &JsonValue) -> Quaternion<f64> {
        Quaternion::new(
            *json[3].get::<f64>().unwrap(),
            *json[0].get::<f64>().unwrap(),
            *json[1].get::<f64>().unwrap(),
            *json[2].get::<f64>().unwrap(),
        )
    }

    fn parse_matrix3(json: &JsonValue) -> Matrix3<f64> {
        let vals = json
            .get::<Vec<_>>()
            .unwrap()
            .iter()
            .map(|v| *v.get::<f64>().unwrap())
            .collect::<Vec<f64>>();
        Matrix3::from_row_slice(&vals)
    }

    fn command(&self, cmd_id: u8, data: &[u8]) -> Result<Vec<u8>> {
        let command = ImuPacket {
            cmd_id,
            data: data.into(),
        }
        .serialize()
        .ok_or(Error::Other("Couldn't get acknowledgement to command"))?;
        write_hid_packet(&self.device, &command)?;
        let packet_size = self.model.imu_packet_size();
        for _ in 0..64 {
            let mut data = vec![0u8; packet_size];
            let result_size = self.device.read_timeout(&mut data, IMU_TIMEOUT)?;
            if result_size == 0 {
                return Err(Error::PacketTimeout);
            }

            if let Some(result) = ImuPacket::deserialize(&data) {
                return Ok(result.data);
            }
        }
        Err(Error::Other("Couldn't get acknowledgement to command"))
    }

    fn start_packet_logging(&mut self, path: &Path) -> Result<()> {
        if self.packet_log.is_some() {
            return Err(Error::Other("Packet logging already started"));
        }

        let imu_config = self.config_json["IMU"]["device_1"]
            .stringify()
            .map_err(|_| Error::Other("Couldn't serialize IMU calibration for packet log"))?;
        let mut packet_log = BufWriter::new(File::create(path)?);
        writeln!(
            packet_log,
            "# {PACKET_LOG_MAGIC}\t{}\t{imu_config}",
            self.model.packet_log_name()
        )?;
        packet_log.flush()?;
        self.packet_log = Some(packet_log);
        Ok(())
    }

    fn stop_packet_logging(&mut self) -> Result<()> {
        if let Some(mut packet_log) = self.packet_log.take() {
            packet_log.flush()?;
        }
        Ok(())
    }

    fn log_packet(&mut self, packet: &[u8]) -> Result<()> {
        if let Some(packet_log) = &mut self.packet_log {
            for byte in packet {
                write!(packet_log, "{byte:02x}")?;
            }
            writeln!(packet_log)?;
            packet_log.flush()?;
        }
        Ok(())
    }

    pub fn read_packet(&mut self) -> Result<GlassesEvent> {
        loop {
            if let Some(event) = self.base.pop_event() {
                return Ok(event);
            }
            let mut packet_data = [0u8; 0x80];
            let data_size = self.device.read_timeout(&mut packet_data, IMU_TIMEOUT)?;
            if data_size == 0 {
                return Err(Error::PacketTimeout);
            }
            self.log_packet(&packet_data[..data_size])?;

            self.base.push_packet(&packet_data[..data_size])?;
            // Else try again
        }
    }
}

#[derive(Default)]
struct XrealAirBase {
    pending_events: VecDeque<GlassesEvent>,
    gyro_bias: Vector3<f32>,
    accelerometer_bias: Vector3<f32>,
}

impl XrealAirBase {
    fn from_calibration(calibration: &JsonValue) -> Result<Self> {
        Ok(Self {
            pending_events: VecDeque::new(),
            gyro_bias: parse_calibration_vector(calibration, "gyro_bias")?,
            accelerometer_bias: parse_calibration_vector(calibration, "accel_bias")?,
        })
    }

    fn pop_event(&mut self) -> Option<GlassesEvent> {
        self.pending_events.pop_front()
    }

    fn push_packet(&mut self, packet_data: &[u8]) -> Result<()> {
        if packet_data.starts_with(&[1, 2]) {
            self.pending_events
                .extend(self.decode_sensor_report(packet_data)?);
        }
        Ok(())
    }

    fn decode_sensor_report(&self, packet_data: &[u8]) -> Result<Vec<GlassesEvent>> {
        let mut ret = Vec::with_capacity(2);
        // TODO: This skips over a 2 byte temperature field that may be useful.
        let mut reader = std::io::Cursor::new(&packet_data[4..]);

        let timestamp = reader.read_u64::<LittleEndian>()? / 1000;
        let gyro_mul = reader.read_u16::<LittleEndian>()? as f32;
        let gyro_div = reader.read_u32::<LittleEndian>()? as f32;
        let gyro_x = reader.read_i24::<LittleEndian>()? as f32;
        let gyro_y = reader.read_i24::<LittleEndian>()? as f32;
        let gyro_z = reader.read_i24::<LittleEndian>()? as f32;
        let gyroscope = Vector3::new(
            // The bias fields do not correspond to the raw fields, but for some reason
            // this looks like the correct zero.
            -(gyro_x * gyro_mul / gyro_div).to_radians() - self.gyro_bias.x,
            (gyro_z * gyro_mul / gyro_div).to_radians() + self.gyro_bias.y,
            (gyro_y * gyro_mul / gyro_div).to_radians() + self.gyro_bias.z,
        );

        let acc_mul = reader.read_u16::<LittleEndian>()? as f32;
        let acc_div = reader.read_u32::<LittleEndian>()? as f32;
        let acc_x = reader.read_i24::<LittleEndian>()? as f32;
        let acc_y = reader.read_i24::<LittleEndian>()? as f32;
        let acc_z = reader.read_i24::<LittleEndian>()? as f32;
        let accelerometer = Vector3::new(
            // The bias fields do not correspond to the raw fields, but for some reason
            // this looks like the correct zero.
            -(acc_x * acc_mul / acc_div) * 9.81 - self.accelerometer_bias.x,
            (acc_z * acc_mul / acc_div) * 9.81 + self.accelerometer_bias.y,
            (acc_y * acc_mul / acc_div) * 9.81 + self.accelerometer_bias.z,
        );

        if let Some(XrealMagnetometerReport {
            magnetic_field,
            sensor_timestamp_nanos,
            fresh: true,
        }) = decode_xreal_magnetometer_report(packet_data)
        {
            // The event API has no transport-metadata channel. Keep using the
            // report's primary device timestamp, as ar-glass-lib does.
            let _sensor_timestamp_nanos = sensor_timestamp_nanos;
            if is_valid_magnetic_observation(&magnetic_field) {
                // Send magnetometer event first so that clients can match the most
                // recent magnetometer event to the most recent accgyro event and not get
                // out of sync. This is necessary because the magnetometer event is
                // optional.
                ret.push(GlassesEvent::Magnetometer {
                    magnetometer: magnetic_field,
                    timestamp,
                });
            }
        }

        // TODO: Replace the XREAL Air 1 magnetometer decoder with a hardware-backed implementation.
        //
        // Recovery workflow (one commit per numbered step):
        // 1. Add object-safe `ARGlasses` methods that start and stop packet logging. Their default
        //    implementations are no-ops.
        // 2. Implement logging for `XrealAir`. The versioned text file starts with the model and the
        //    sanitized `IMU.device_1` calibration object; every remaining line is one exact HID packet
        //    encoded as lowercase hexadecimal.
        // 3. Add a 60-second Air 1 logging example and capture a continuous three-axis calibration dance.
        // 4. Add `XrealAirReplay`, which validates the log and replays its packets cyclically.
        // 5. Move all sensor-report parsing and event queuing into a shared `XrealAirBase` used by the live
        //    and replay transports.
        // 6. Convert the `read_sensors` calibration experiment into a replay integration test that reports
        //    every `CalibrationQuality` factor.
        // 7. Rewrite only the version-2 magnetometer path. Treat ar-glass-lib, Monado, and XRLinuxDriver as
        //    competing hypotheses rather than ground truth; reject stale, non-finite, zero-divisor, and
        //    zero-norm magnetic observations without changing accelerometer or gyroscope output.
        // 8. Require `fitness`, `radial_fitness`, and `gravity_fitness` to remain at least 0.8 for the final
        //    five seconds of fresh observations after calibration becomes usable, then remove this TODO.

        // TODO: Check checksum
        ret.push(GlassesEvent::AccGyro {
            accelerometer,
            gyroscope,
            timestamp,
        });
        Ok(ret)
    }
}

#[derive(Debug, Default)]
struct McuPacket {
    cmd_id: u16,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct McuRawPacket {
    head: u8,
    checksum: u32,
    length: u16,
    request_id: u32,
    timestamp: u32,
    cmd_id: u16,
    reserved: [u8; 5],
    data: [u8; 42],
}

unsafe impl bytemuck::Zeroable for McuRawPacket {}
unsafe impl bytemuck::Pod for McuRawPacket {}

impl McuPacket {
    fn deserialize(data: &[u8; 0x40]) -> Option<McuPacket> {
        let raw_packet: &McuRawPacket = bytemuck::cast_ref(data);
        if raw_packet.head != 0xfd {
            return None;
        }
        // TODO: maybe check CRC?
        Some(McuPacket {
            cmd_id: raw_packet.cmd_id,
            data: raw_packet.data[0..(raw_packet.length as usize - 17)].into(),
        })
    }

    fn serialize(&self) -> Option<[u8; 0x40]> {
        let mut data = [0u8; 42];
        data[0..self.data.len()].copy_from_slice(&self.data);
        let mut raw_packet = McuRawPacket {
            head: 0xfd,
            checksum: 0,
            length: self.data.len() as u16 + 17,
            request_id: 0x1337,
            timestamp: 0x0,
            cmd_id: self.cmd_id,
            reserved: Default::default(),
            data,
        };
        raw_packet.checksum =
            crc32_adler(&bytemuck::bytes_of(&raw_packet)[5..(5 + raw_packet.length as usize)]);
        Some(bytemuck::cast(raw_packet))
    }
}

#[derive(Debug, Default)]
struct ImuPacket {
    cmd_id: u8,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct ImuPacketHeader {
    head: u8,
    checksum: u32,
    length: u16,
    cmd_id: u8,
}

unsafe impl bytemuck::Zeroable for ImuPacketHeader {}
unsafe impl bytemuck::Pod for ImuPacketHeader {}

const IMU_HEADER_SIZE: usize = std::mem::size_of::<ImuPacketHeader>();

impl ImuPacket {
    fn deserialize(data: &[u8]) -> Option<ImuPacket> {
        if data.len() < IMU_HEADER_SIZE {
            return None;
        }

        let header: &ImuPacketHeader = bytemuck::from_bytes(&data[..IMU_HEADER_SIZE]);
        if header.head != 0xaa {
            return None;
        }

        // length includes cmd_id (1 byte) + checksum bytes (2 bytes) + data
        let data_len = (header.length as usize).saturating_sub(3);
        let data_end = IMU_HEADER_SIZE + data_len;

        if data_end > data.len() {
            return None;
        }

        // TODO: maybe check CRC?
        Some(ImuPacket {
            cmd_id: header.cmd_id,
            data: data[IMU_HEADER_SIZE..data_end].into(),
        })
    }

    fn serialize(&self) -> Option<[u8; 0x40]> {
        let mut result = [0u8; 0x40];
        let header = ImuPacketHeader {
            head: 0xaa,
            checksum: 0,
            length: self.data.len() as u16 + 3,
            cmd_id: self.cmd_id,
        };
        result[..IMU_HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
        result[IMU_HEADER_SIZE..IMU_HEADER_SIZE + self.data.len()].copy_from_slice(&self.data);
        let checksum = crc32_adler(&result[5..(5 + header.length as usize)]);
        result[1..5].copy_from_slice(&checksum.to_le_bytes());
        Some(result)
    }
}

#[cfg(not(target_os = "android"))]
fn open_xreal_air() -> Result<(AirModel, HidDevice, HidDevice)> {
    let hidapi = HidApi::new()?;

    // First find the model by checking MCU interfaces
    let mut found_model: Option<AirModel> = None;
    let mut mcu_device: Option<HidDevice> = None;

    for device in hidapi.device_list() {
        if device.vendor_id() != XREAL_VID {
            continue;
        }
        let pid = device.product_id();
        let model = match AirModel::try_from(pid) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if device.interface_number() == model.mcu_interface() {
            found_model = Some(model);
            mcu_device = Some(device.open_device(&hidapi)?);
            break;
        }
    }

    let model = found_model.ok_or(Error::NotFound)?;
    let mcu = mcu_device.ok_or(Error::NotFound)?;

    // Now open the IMU interface for this model
    let imu_iface = model.imu_interface();
    let mut imu_device: Option<HidDevice> = None;

    for device in hidapi.device_list() {
        if device.vendor_id() != XREAL_VID {
            continue;
        }
        let pid = device.product_id();
        if AirModel::try_from(pid).is_err() {
            continue;
        }
        if device.interface_number() == imu_iface {
            imu_device = Some(device.open_device(&hidapi)?);
            break;
        }
    }

    let imu = imu_device.ok_or(Error::NotFound)?;

    Ok((model, mcu, imu))
}

#[cfg(target_os = "windows")]
fn write_hid_packet(device: &HidDevice, payload: &[u8; 0x40]) -> Result<()> {
    // Windows HID writes need the leading report ID byte even for unnumbered reports.
    // The Air replies on interfaces 3/4 once the payload is sent as [0x00 | 64-byte packet].
    let mut report = [0u8; 0x41];
    report[1..].copy_from_slice(payload);
    device.write(&report)?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn write_hid_packet(device: &HidDevice, payload: &[u8; 0x40]) -> Result<()> {
    device.write(payload)?;
    Ok(())
}

#[cfg(test)]
#[path = "xreal_air_tests.rs"]
mod xreal_air_tests;
