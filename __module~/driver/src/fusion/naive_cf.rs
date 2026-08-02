//!
//! NaiveCF is a simple sensor fusion algorithm that uses a complementary filter.
//!
//! complementary filter with very simple update algorithm:
//!
//! (
//!     assuming:
//!     (S, S-) = current & previous estimated state
//!     d~S1 = rate sensor reading (e.g. gyro), high frequency, high drift, dead reckoning
//!    ~S2 = state sensor reading (e.g. grav/acc, mag), low frequency, high noise, low drift
//!    d_t1 = time elapse since last rate sensor sampling
//!)
//!
//!S = S- + ratio * d~S1 * d_t1 + (1-ratio) * (~S2 - S-)
//!  = ratio * (S- + d~S1 * d_t1) + (1-ratio) * ~S2
//!
//!this implies:
//!
//!- the algorithm natively support state sensor(s) with different sampling frequency, incomplete reading,
//!or unreliable reading, as ~S2 variable is merely an optional correction
//!- the interpolation between ~S2 and the first term doesn't need to be linear or additive,
//!e.g. 3D angular interpolation is multiplicative
//!- the ratio can be adjusted based on quality & frequency of state sensor(s)
//!
//!most glasses have acc & grav/acc readings in 1 bundle, but I prefer not using this assumption and still update them independently
//!
//! # example:
//!  TODO: fill

use nalgebra::{UnitQuaternion, Vector3};

use super::{rub_to_frd, Correction, Fusion, FusionState, NineAxis};
use crate::{ARGlasses, Error, GlassesEvent};

type Result<T> = std::result::Result<T, Error>;

pub struct NaiveCF {
    pub state: FusionState,
    pub prev_gyro: (Vector3<f32>, u64), //FRD
}

impl NaiveCF {
    pub fn new(glasses: Box<dyn ARGlasses>) -> Result<Self> {
        //let attitude = ;
        //let prev_gyro = ;
        let mut fusion = NaiveCF {
            state: FusionState::new(glasses),
            prev_gyro: (Vector3::zeros(), 0),
        };

        loop {
            //wait for the first non-zero gyro reading
            let next_event = fusion.next_event();
            match next_event {
                GlassesEvent::AccGyro {
                    accelerometer: _,
                    gyroscope,
                    timestamp,
                } => {
                    //if gyroscope != Vector3::zeros() {
                    fusion.prev_gyro = (gyroscope, timestamp);
                    return Ok(fusion);
                    //}
                }
                _ => {}
            }
        }
    }

    ///read until next valid event. Blocks.
    fn next_event(&mut self) -> GlassesEvent {
        loop {
            match self.state.glasses.read_event() {
                Ok(event) => return event,
                Err(e) => {
                    println!("Error reading event: {}", e);
                }
            }
        }
    }

    const BASE_GRAV_RATIO: f32 = 0.005;
    //const BASE_GRAV_RATIO: f32 = 0.0; //no grav
    // const BASE_GRAV_RATIO: f32 = 1.0; //absolute correction, no gyro

    const BASE_MAG_RATIO: f32 = 0.1;

    const GYRO_SPEED_IN_TIMESTAMP_FACTOR: f32 = 1000.0 * 1000.0; //microseconds

    const REGRESS_ROLL_FACTOR: f32 = 0.05; // TODO: should be zero

    const G_ACC_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);
    //const NORTH_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, -1.0);

    //CAUTION: right-multiplication means rotation, unconventionally

    fn integrate_gyro(&mut self, gyro_rub: &Vector3<f32>, t: u64) -> () {
        let gyro = rub_to_frd(gyro_rub);

        let d_t1 = t - self.prev_gyro.1;
        let d_t1_f = d_t1 as f32 / Self::GYRO_SPEED_IN_TIMESTAMP_FACTOR;
        let d_s1_t1 = d_t1_f * gyro;

        let increment = UnitQuaternion::from_euler_angles(d_s1_t1.x, d_s1_t1.y, d_s1_t1.z);

        // self.attitude = (increment.inverse() * self.attitude.inverse()).inverse();
        self.state.attitude = self.state.attitude * increment;
        self.state.corrections.gyro.record(increment.angle());

        self.prev_gyro = (gyro, t);
    }

    fn integrate_acc(&mut self, acc_rub: &Vector3<f32>, _t: u64) -> () {
        let acc = rub_to_frd(acc_rub);

        if acc.norm() < 1.0 {
            return; //almost in free fall, or acc disabled, do not correct
        }

        let attitude = &self.state.attitude;
        // let acc_inv = Vector3::new(-acc.x, -acc.y, acc.z);

        let correction_opt = Self::get_correction(&acc, &attitude.inverse(), Self::BASE_GRAV_RATIO);

        match correction_opt {
            Some(correction_inv) => {
                let correction = correction_inv.inverse();
                self.state.corrections.acc.record(correction.angle());

                // self.attitude = (correction_inv * attitude.inverse()).inverse();
                self.state.attitude = attitude * correction;
            }
            None => {
                // opposite direction, don't know how to correct
            }
        }
    }

    pub(super) fn integrate_regress_roll(&mut self) -> () {
        if !Self::REGRESS_ROLL_FACTOR.is_finite() || (Self::REGRESS_ROLL_FACTOR <= 0.0) {
            return;
        }

        let no_roll_factor = Self::REGRESS_ROLL_FACTOR.clamp(0.0, 1.0);
        let (roll, pitch, yaw) = self.state.attitude.euler_angles();
        let corrected_roll = roll * (1.0 - no_roll_factor);
        self.state.attitude = UnitQuaternion::from_euler_angles(corrected_roll, pitch, yaw);
    }

    pub(super) fn integrate_mag(
        &mut self,
        mag_rub: &Vector3<f32>,
        calibration_use_gravity: bool,
        horizontal_only: bool,
        t: u64,
    ) -> () {
        let mag_raw = rub_to_frd(mag_rub); // reading is always muT (microTesla) pointing to north

        let gravity_hint: Option<Vector3<f32>> = if calibration_use_gravity {
            // gravity direction is already estimated by the acc complementary filter
            Some(self.state.attitude.inverse() * Self::G_ACC_FRD)
        } else {
            None
        };
        let mag_corrected: Vector3<f32> = match self
            .state
            .magCalibrator
            .evaluate_correct(mag_raw, gravity_hint, t)
            .ok()
            .and_then(|result| result.direction)
        {
            Some(r) => r,
            None => return,
        };

        // near the magnetic poles the field is almost vertical and carries no usable heading
        if mag_corrected.norm_squared() < 0.01 {
            return;
        }

        let mag_normalised = mag_corrected.normalize(); // TODO: is it necessary?

        let attitude = &self.state.attitude;
        let north_frd = Vector3::new(1.0, 0.0, 0.0);

        let mag_effective: Vector3<f32> = if horizontal_only {
            // magnetic north dips below the horizon by a location-dependent inclination
            // angle; correcting against the full field vector would tip the level the acc
            // filter already maintains, so only the field's horizontal component is compared.
            // both vectors are then perpendicular to estimated up, making the correction a
            // pure heading rotation about the up axis.
            let up_body = attitude.inverse() * Self::G_ACC_FRD.normalize();
            mag_normalised - mag_normalised.dot(&up_body) * up_body
        } else {
            mag_normalised
        };

        let estimated_north = attitude.inverse() * north_frd;
        let correction_opt = UnitQuaternion::scaled_rotation_between(
            &estimated_north,
            &mag_effective,
            Self::BASE_MAG_RATIO,
        );

        match correction_opt {
            Some(correction_inv) => {
                let correction = correction_inv.inverse();
                self.state.corrections.mag.record(correction.angle());
                self.state.attitude = attitude * correction;
            }
            None => {
                // opposite direction, don't know how to correct
            }
        }
    }

    pub fn get_correction(
        acc: &Vector3<f32>,
        rotation: &UnitQuaternion<f32>,
        scale: f32,
    ) -> Option<UnitQuaternion<f32>> {
        let uncorrected = rotation * Self::G_ACC_FRD.normalize();

        let scaled_opt =
            UnitQuaternion::scaled_rotation_between(&uncorrected, &acc.normalize(), scale);

        // let rotation_opt = Self::get_rotation(acc, rotation);

        // let scaled_opt = match rotation_opt {
        //     Some(correction) => {
        //         let scaled_axis = correction.scaled_axis();
        //         let scaled = UnitQuaternion::from_scaled_axis(scaled_axis * scale);
        //         Some(scaled)
        //     }
        //     None => None,
        // };

        // let scaled_opt = match rotation_opt {
        //     Some(correction) => {
        //         UnitQuaternion::try_slerp(&UnitQuaternion::identity(), &correction, scale, 0.0)
        //     }
        //     None => None,
        // };

        scaled_opt
    }

    // ponytail: kept for temporary debugging; remove together with the rest of the rotation helpers
    #[allow(dead_code)]
    pub fn get_rotation(
        acc: &Vector3<f32>,
        rotation: &UnitQuaternion<f32>,
    ) -> Option<UnitQuaternion<f32>> {
        Self::get_rotation_raw(acc, rotation)
    }

    #[allow(dead_code)]
    fn get_rotation_raw(
        acc: &Vector3<f32>,
        rotation: &UnitQuaternion<f32>,
    ) -> Option<UnitQuaternion<f32>> {
        let uncorrected = rotation * Self::G_ACC_FRD;
        let correction_opt = UnitQuaternion::scaled_rotation_between(&uncorrected, &acc, 1.0);
        correction_opt
    }

    #[allow(dead_code)]
    fn get_rotation_verified(
        acc: &Vector3<f32>,
        rotation: &UnitQuaternion<f32>,
    ) -> Option<UnitQuaternion<f32>> {
        let raw = Self::get_rotation_raw(acc, rotation);
        match raw {
            Some(correction) => {
                //round-trip verification

                let corrected = correction * rotation;

                let should_be_zero = Self::get_rotation_raw(acc, &corrected).unwrap().angle();

                if should_be_zero > 0.001 {
                    println!("residual={}", should_be_zero);
                    println!(
                        "compute: {}, {} => {}",
                        acc.transpose(),
                        rotation,
                        correction
                    );

                    {
                        let norm = rotation.norm();
                        assert!(norm > 0.999 && norm < 1.001, "norm={}", norm);
                    }

                    {
                        let reconstructed = UnitQuaternion::from_axis_angle(
                            &rotation.axis().unwrap(),
                            rotation.angle(),
                        );

                        assert!((rotation * reconstructed.inverse()).angle() < 0.001);

                        assert!(
                            (rotation * Self::G_ACC_FRD.normalize()
                                - reconstructed * Self::G_ACC_FRD.normalize())
                            .norm()
                                < 0.01
                        )
                    }

                    {
                        let again = Self::get_rotation_raw(acc, rotation);
                        assert!(raw == again)
                    }

                    {
                        // verify rotation
                        let inv = rotation.inverse();
                        assert!((inv * rotation).angle() < 0.001);
                        assert!((rotation * inv).angle() < 0.001);
                    }

                    {
                        // verity acc
                        let q = UnitQuaternion::scaled_rotation_between(&Self::G_ACC_FRD, acc, 1.0)
                            .unwrap();

                        let round1 = (q * Self::G_ACC_FRD.normalize() - acc.normalize()).norm();
                        assert!(round1 < 0.001, "round1={}", round1);

                        let round2 =
                            (q.inverse() * acc.normalize() - Self::G_ACC_FRD.normalize()).norm();
                        assert!(round2 < 0.001, "round2={}", round2);
                    }
                }

                raw
            }
            None => raw,
        }
    }

    fn renormalize(&mut self) {
        // self.attitude.renormalize_fast(); // TODO: switch to it after rigorous testing
        self.integrate_regress_roll();
        self.state.attitude.renormalize();
    }
}

//unsafe impl Sync for NaiveCF {}

impl Fusion for NaiveCF {
    fn glasses(&mut self) -> &mut Box<dyn ARGlasses> {
        &mut self.state.glasses
    }

    fn attitude_quaternion(&self) -> UnitQuaternion<f32> {
        self.state.attitude
    }

    fn corrections(&self) -> NineAxis<Correction> {
        self.state.corrections
    }

    fn update(&mut self) -> () {
        let event = self.next_event();
        match event {
            GlassesEvent::AccGyro {
                accelerometer,
                gyroscope,
                timestamp,
            } => {
                // self.integrate_gyro(&gyroscope, timestamp);
                // self.integrate_acc(&accelerometer, timestamp);
                // self.renormalize();
            }

            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                self.integrate_mag(&magnetometer, false, false, timestamp);
                // self.integrate_mag(&magnetometer, true, true, timestamp); TODO: use this
                self.renormalize();
            }
            _ => {
                // TODO: handle KeyPress signal
            }
        }
    }
}
