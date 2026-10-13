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

- [x]  Re-tune the mag/dip blend ratios after the dip wiring

    - **Summary:** Re-tuned jointly: `BASE_MAG_RATIO` 0.005 -> 0.00125 and `BASE_DIP_RATIO` 0.02 -> 0.05.
      The cold-start level transient is the acc-fought level content of the correction's chase of the
      calibrator's converging azimuthal error, so it scales with the mag/grav blend balance: on the
      gentle-motion SimMotion profile the cold-start scored half's mean velocity error measures
      1.54 m/s at 0.005, 0.81 at 0.0025, and 0.38 at 0.00125 — the settled floor's scale — while the
      faster dip state hands the field's vertical residual to `mag_dip_sin` before the correction can
      tip the level with it (0.42 -> 0.38 m/s alongside the mag re-tune), and the estimated/true
      speed ratio returns from 4.3 to 1.25. The heading lock trades as accepted: a 30 deg heading
      step's e-fold time grows from a measured ~9 s to ~33 s, converging to the same ~1 deg floor,
      and the default 20 RPM profile is unchanged (mean |v_est| 0.63 m/s, halves 0.76 -> 0.55 m/s).
    - **Position:** `src/fusion/_3dof/naive_cf_core.rs (BASE_MAG_RATIO, BASE_DIP_RATIO)` (the two
      blend-ratio constants' doc comments carry the tuning rationale; the original
      `retune_ratios_after_dip_wiring` block comment was replaced by the fix)
    - **Unit test:** `src/fusion/_6dof/naive_cf_test.rs (velocity_tracks_gentle_motion)` — its
      `MAX_COLD_START_ABS_ERROR` bound shrank 2.0 -> 0.45 (measured 1.54 -> 0.38 m/s) onto the
      settled bound, and `MAGNITUDE_RATIO_BAND` tightened (0.4, 6.0) -> (0.6, 2.0) on the
      measured 1.25.
