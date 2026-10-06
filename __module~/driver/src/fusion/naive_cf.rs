//!
//! NaiveCF is a simple sensor fusion algorithm that uses a complementary filter.
//!
//! Complementary filter with a very simple update algorithm. With $S$ and $S^{-}$ the current and
//! previous estimated state, $d\tilde{S}_1$ a rate sensor reading (e.g. gyroscope: high frequency, high
//! drift, dead reckoning), $\tilde{S}_2$ a state sensor reading (e.g. gravity/accelerometer, magnetometer:
//! low frequency, high noise, low drift), $dt_1$ the time elapsed since the last rate sensor sample, and
//! $\mathrm{ratio}$ the blend ratio:
//!
//! $$S = \mathrm{ratio}\, (S^{-} + d\tilde{S}_1\, dt_1) + (1 - \mathrm{ratio})\, \tilde{S}_2$$
//!
//! This implies:
//!
//! - The algorithm natively supports state sensor(s) with different sampling frequency, incomplete
//!   reading, or unreliable reading, as the $\tilde{S}_2$ variable is merely an optional correction.
//! - The interpolation between $\tilde{S}_2$ and the rate-integrated prediction doesn't need to be linear
//!   or additive, e.g. 3D angular interpolation is multiplicative.
//! - The ratio can be adjusted based on quality and frequency of the state sensor(s).
//!
//! Most glasses have acc & grav/acc readings in 1 bundle, but I prefer not using this assumption and still
//! update them independently.
//!
//! # example:
//!  TODO: fill

use nalgebra::{UnitQuaternion, Vector3};

use super::{rub_to_frd, Consistency, Fusion, FusionState};
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

    const BASE_GRAV_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.005;
    //const BASE_GRAV_RATIO: f32 = 0.0; //no grav
    // const BASE_GRAV_RATIO: f32 = 1.0; //absolute correction, no gyro

    const BASE_MAG_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.1;

    /// Blend ratio $1 - \mathrm{ratio}$ of the per-sample dip update. The
    /// vertical field component feeds the dip state instead of the
    /// attitude, so its innovation is yaw-invariant and the remaining
    /// disturbance is the roll/pitch error the acc filter keeps small;
    /// the ratio can therefore be small, tracking the slow geographic drift
    /// of the true dip rather than chasing single noisy readings.
    const BASE_DIP_RATIO /*$1 - \mathrm{ratio}$*/: f32 = 0.02;

    /// Minimum norm of the calibrated field's horizontal component, as a
    /// fraction of the unit field, below which the heading update is gated.
    /// A field closer than $\arcsin(0.1) \approx 5.7^{\circ}$ to vertical
    /// (near the magnetic poles) carries almost no azimuth information, so
    /// its horizontal noise would rotate the attitude about the down axis
    /// without actual heading evidence.
    const MIN_HEADING_FIELD_NORM: f32 = 0.1;

    const G_ACC_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);
    /// Unit down axis of the FRD world frame.
    const DOWN_FRD: Vector3<f32> = Vector3::new(0.0, 0.0, 1.0);
    /// Unit magnetic-north axis of the FRD world frame; the field's
    /// horizontal component points here whatever the dip angle.
    const NORTH_FRD: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);

    //CAUTION: right-multiplication means rotation, unconventionally

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
                let _ = self
                    .state
                    .consistency
                    .sources
                    .acc
                    .record_scaled(correction.angle(), Self::BASE_GRAV_RATIO);

                // self.attitude = (correction_inv * attitude.inverse()).inverse();
                self.state.attitude = attitude * correction;
            }
            None => {
                // opposite direction, don't know how to correct
            }
        }
    }

    /// Magnetometer update with the magnetic dip $\delta$ as part of the
    /// estimated state: the FRD reference field is
    /// $(\cos\delta, 0, \sin\delta)$, positive below the horizon, so the
    /// reading is never corrected towards a purely horizontal north. The
    /// single vector observation cannot fix attitude and dip jointly — a
    /// level (roll/pitch) error and a dip error produce the same vertical
    /// residual — so the update is partitioned instead of minimized
    /// jointly: the estimated vertical component refines only $\delta$
    /// (leaving level to the acc filter), and only the horizontal
    /// component refines the heading (leaving the field's vertical split
    /// to $\delta$).
    pub(super) fn integrate_mag(
        &mut self,
        mag_rub: &Vector3<f32>,
        calibration_use_gravity: bool,
        t: u64,
    ) -> () {
        let mag_raw = rub_to_frd(mag_rub); // reading is always muT (microTesla) pointing to north

        let gravity_hint: Option<Vector3<f32>> = if calibration_use_gravity {
            // gravity direction is already estimated by the acc complementary filter
            Some(self.state.attitude.inverse() * Self::G_ACC_FRD)
        } else {
            None
        };
        let result = match self
            .state
            .mag_calibrator
            .evaluate_correct(mag_raw, gravity_hint, t)
        {
            Ok(result) => result,
            Err(_) => return,
        };
        // `direction` is already the normalized calibrated unit vector.
        let mag_normalised: Vector3<f32> = match result.direction {
            Some(direction) => direction,
            None => return,
        };

        // Seed the dip once from the calibrator's learned dip projection:
        // by first publication the gravity surrogate's
        // $\kappa / (\gamma r)$ has already seen the co-rotated gravity
        // hints, so its seed skips the cold-start transient of the
        // per-sample refinement below. The hint passed above is the
        // accelerometer direction, which points up in FRD, giving
        // `dip_sin` $= g^T m = -\sin\delta$.
        if self.state.mag_dip_sin.is_none() {
            if let Some(dip_sin) = result.dip_sin {
                self.state.mag_dip_sin = Some(-dip_sin.clamp(-1.0, 1.0));
            }
        }

        let attitude = &self.state.attitude;
        let down_body = attitude.inverse() * Self::DOWN_FRD;

        // Dip step: the vertical component of the measured field estimates
        // $\sin\delta$, so the state is tracked directly as $\sin\delta$
        // and the update is linear in it. The vertical component is
        // invariant to heading error (a rotation about the down axis
        // preserves it), so even the large yaw transient right after boot
        // cannot flip the update's sign; only roll/pitch error propagates,
        // which the acc filter holds small relative to `BASE_DIP_RATIO`'s
        // pull.
        let vertical = mag_normalised.dot(&down_body);
        let dip_sin = self.state.mag_dip_sin.unwrap_or(0.0)
            + Self::BASE_DIP_RATIO * (vertical - self.state.mag_dip_sin.unwrap_or(0.0));
        self.state.mag_dip_sin = Some(dip_sin.clamp(-1.0, 1.0));

        // Heading step: only the field's horizontal component (a pure
        // north reading) is compared against estimated north. Both vectors
        // are perpendicular to estimated down — north as a world
        // horizontal, the measured component by construction — so the
        // correction is a pure heading rotation about the down axis and
        // can never tip the level the acc filter maintains, whatever the
        // current dip estimate.
        let mag_horizontal = mag_normalised - vertical * down_body;
        if mag_horizontal.norm() < Self::MIN_HEADING_FIELD_NORM {
            return;
        }

        let estimated_north = attitude.inverse() * Self::NORTH_FRD;
        let correction_opt = UnitQuaternion::scaled_rotation_between(
            &estimated_north,
            &mag_horizontal,
            Self::BASE_MAG_RATIO,
        );

        match correction_opt {
            Some(correction_inv) => {
                let correction = correction_inv.inverse();
                let _ = self
                    .state
                    .consistency
                    .sources
                    .mag
                    .record_scaled(correction.angle(), Self::BASE_MAG_RATIO);
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
        scale /*$1 - \mathrm{ratio}$*/: f32,
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

    // #[allow(dead_code)]
    // pub fn get_rotation(
    //     acc: &Vector3<f32>,
    //     rotation: &UnitQuaternion<f32>,
    // ) -> Option<UnitQuaternion<f32>> {
    //     Self::get_rotation_raw(acc, rotation)
    // }

    // #[allow(dead_code)]
    // fn get_rotation_raw(
    //     acc: &Vector3<f32>,
    //     rotation: &UnitQuaternion<f32>,
    // ) -> Option<UnitQuaternion<f32>> {
    //     let uncorrected = rotation * Self::G_ACC_FRD;
    //     let correction_opt = UnitQuaternion::scaled_rotation_between(&uncorrected, &acc, 1.0);
    //     correction_opt
    // }

    // #[allow(dead_code)]
    // fn get_rotation_verified(
    //     acc: &Vector3<f32>,
    //     rotation: &UnitQuaternion<f32>,
    // ) -> Option<UnitQuaternion<f32>> {
    //     let raw = Self::get_rotation_raw(acc, rotation);
    //     match raw {
    //         Some(correction) => {
    //             //round-trip verification
    //
    //             let corrected = correction * rotation;
    //
    //             let should_be_zero = Self::get_rotation_raw(acc, &corrected).unwrap().angle();
    //
    //             if should_be_zero > 0.001 {
    //                 println!("residual={}", should_be_zero);
    //                 println!(
    //                     "compute: {}, {} => {}",
    //                     acc.transpose(),
    //                     rotation,
    //                     correction
    //                 );
    //
    //                 {
    //                     let norm = rotation.norm();
    //                     assert!(norm > 0.999 && norm < 1.001, "norm={}", norm);
    //                 }
    //
    //                 {
    //                     let reconstructed = UnitQuaternion::from_axis_angle(
    //                         &rotation.axis().unwrap(),
    //                         rotation.angle(),
    //                     );
    //
    //                     assert!((rotation * reconstructed.inverse()).angle() < 0.001);
    //
    //                     assert!(
    //                         (rotation * Self::G_ACC_FRD.normalize()
    //                             - reconstructed * Self::G_ACC_FRD.normalize())
    //                         .norm()
    //                             < 0.01
    //                     )
    //                 }
    //
    //                 {
    //                     let again = Self::get_rotation_raw(acc, rotation);
    //                     assert!(raw == again)
    //                 }
    //
    //                 {
    //                     // verify rotation
    //                     let inv = rotation.inverse();
    //                     assert!((inv * rotation).angle() < 0.001);
    //                     assert!((rotation * inv).angle() < 0.001);
    //                 }
    //
    //                 {
    //                     // verity acc
    //                     let q = UnitQuaternion::scaled_rotation_between(&Self::G_ACC_FRD, acc, 1.0)
    //                         .unwrap();
    //
    //                     let round1 = (q * Self::G_ACC_FRD.normalize() - acc.normalize()).norm();
    //                     assert!(round1 < 0.001, "round1={}", round1);
    //
    //                     let round2 =
    //                         (q.inverse() * acc.normalize() - Self::G_ACC_FRD.normalize()).norm();
    //                     assert!(round2 < 0.001, "round2={}", round2);
    //                 }
    //             }
    //
    //             raw
    //         }
    //         None => raw,
    //     }
    // }

    fn renormalize(&mut self) {
        // self.attitude.renormalize_fast(); // TODO: switch to it after rigorous testing
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

    fn consistency(&self) -> Consistency {
        self.state.consistency
    }

    fn update(&mut self) -> () {
        let event = self.next_event();
        match event {
            GlassesEvent::AccGyro {
                accelerometer,
                gyroscope,
                timestamp,
            } => {
                // dead-reckoning increment magnitude, kept as a sensor-health signal
                let gyro = rub_to_frd(&gyroscope);
                let dt_seconds = (timestamp - self.prev_gyro.1) as f32 * 1e-6;
                if dt_seconds > 0.0 {
                    let _ = self
                        .state
                        .consistency
                        .sources
                        .gyro
                        .record((gyro * dt_seconds).norm());
                }
                self.prev_gyro = (gyro, timestamp);

                self.integrate_acc(&accelerometer, timestamp);
                self.renormalize();
            }

            GlassesEvent::Magnetometer {
                magnetometer,
                timestamp,
            } => {
                // The ACC-derived gravity hint keeps the calibrator's
                // preconditioned gravity surrogate anchored to the exact
                // magnetic dip, whose learned projection seeds the fusion
                // dip state once at publication; afterwards integrate_mag
                // refines the dip from the field's vertical component
                // itself. The dip/heading partition keeps the magnetometer
                // away from the level estimate in both roles.
                self.integrate_mag(&magnetometer, true, timestamp);
                self.renormalize();
            }
            _ => {
                // TODO: handle KeyPress signal
            }
        }
    }
}
