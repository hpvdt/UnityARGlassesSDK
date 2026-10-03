use crate::fusion::{Fusion, NineAxis};
use std::fmt;

/// Last and averaged correction magnitudes for a sensor.
#[derive(Clone, Copy, Debug)]
pub struct Correction {
    /// Most recent correction magnitude in radians.
    pub prev: f32, // previous
    /// Exponential moving average of correction magnitude in radians.
    pub avg: f32, // average
}

impl Correction {
    /// Exponential averaging decay rate.
    pub const AVG_DECAY: f32 = 0.90;

    fn new() -> Self {
        Self {
            prev: 0.0,
            avg: 0.0,
        }
    }

    pub fn record(&mut self, correction: f32) -> () {
        self.prev = correction;
        self.avg = self.avg * Self::AVG_DECAY + correction * (1.0 - Self::AVG_DECAY);
    }
}

struct Inconsistency {
    pub innovations: NineAxis<Correction>,
}

impl NineAxis<Correction> {
    fn total_avg(&self) -> f32 {
        self.acc.avg + self.gyro.avg + self.mag.avg
    }
}

impl Default for Correction {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Correction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prev={:10.7}, avg={:10.7}", self.prev, self.avg)
    }
}

/// Non-overridable fusion inconsistency computation.
pub trait FusionInconsistency {
    /// use FRD frame as error in Quaternion is multiplicative & is over-defined
    fn inconsistency(&self) -> f32;
}

impl<T: Fusion + ?Sized> crate::fusion::FusionInconsistency for T {
    fn inconsistency(&self) -> f32 {
        self.corrections().total_avg()
    }
}
