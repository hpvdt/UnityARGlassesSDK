use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::{Isometry3, Vector3};

use crate::{ARGlasses, DisplayMode, GlassesEvent, Side};

pub static GRAVITY_UP: Vector3<f32> = Vector3::new(0.0, 9.81, 0.0);
pub static ZERO: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);

pub struct Dummy {}

impl Dummy {}

impl ARGlasses for Dummy {
    fn serial(&mut self) -> crate::Result<String> {
        Ok(String::from("dummy!"))
    }

    fn read_event(&mut self) -> crate::Result<GlassesEvent> {
        let result = GlassesEvent::AccGyro {
            accelerometer: GRAVITY_UP,
            gyroscope: ZERO,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        Ok(result)
    }

    fn get_display_mode(&mut self) -> crate::Result<DisplayMode> {
        Ok(DisplayMode::SameOnBoth)
    }

    fn set_display_mode(&mut self, _display_mode: DisplayMode) -> crate::Result<()> {
        Ok(())
    }

    fn display_fov(&self) -> f32 {
        24.0f32.to_radians()
    }

    fn imu_to_display_matrix(&self, _side: Side, _ipd: f32) -> Isometry3<f64> {
        todo!()
    }

    fn name(&self) -> &'static str {
        "dummy"
    }

    fn display_delay(&self) -> u64 {
        0
    }
}
