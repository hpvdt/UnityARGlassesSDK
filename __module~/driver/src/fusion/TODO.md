## High severity

## Medium severity

- [x]  Wire the learned magnetic dip into the attitude correction

    - **Summary:** The magnetometer attitude correction partitions the dip out of the update: it compares only the
      field's horizontal component against estimated north, so the learned dip state `mag_dip_sin` never influences
      the attitude, and the correction cannot move the level the acc filter maintains.
      Fixed by comparing the full calibrated field against the dipped reference field built from `mag_dip_sin`,
      dropping the horizontal projection and its azimuth gate, so both the yaw and the level parts of the correction
      share the calibrator's usable-direction gate.
    - **Position:** `src/fusion/_3dof/naive_cf_core.rs (integrate_mag)` (the attitude step; the original
      `attitude_correction_ignores_learned_dip` block comment was replaced by the fix)
    - **Unit test:** `src/fusion/_3dof/naive_cf_test.rs (update_mag_correction_tips_level_through_learned_dip)`

- [ ]  Re-tune the mag/dip blend ratios after the dip wiring

    - **Summary:** Wiring the dip into the attitude correction made the level absorb the calibrator's directional
      error while the mag calibration converges: on the gentle-motion SimMotion profile the cold-start scored half's
      mean velocity error inflates to a measured 1.54 m/s (pre-wiring whole-window mean 0.29 m/s) and the
      estimated/true speed ratio to 4.3, settling back to the pre-wiring floor once the calibration, dip, and level
      converge. `BASE_MAG_RATIO` and `BASE_DIP_RATIO` were tuned for the partitioned, level-inert correction and must
      be re-tuned jointly, trading heading lock time against the cold-start level transient.
    - **Position:** `src/fusion/_3dof/naive_cf_core.rs (retune_ratios_after_dip_wiring)`
    - **Unit test:** `src/fusion/_6dof/naive_cf_test.rs (velocity_tracks_gentle_motion)` — its
      `MAX_COLD_START_ABS_ERROR` bound (2.0 on the measured 1.54) is the envelope the re-tuning must shrink
      towards the settled 0.45 bound.
