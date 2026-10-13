## High severity

## Medium severity

- [ ]  Wire the learned magnetic dip into the attitude correction

    - **Summary:** The magnetometer attitude correction partitions the dip out of the update: it compares only the
      field's horizontal component against estimated north, so the learned dip state `mag_dip_sin` never influences
      the attitude, and the correction cannot move the level the acc filter maintains.
    - **Position:** `src/fusion/_3dof/naive_cf_core.rs (attitude_correction_ignores_learned_dip)`
    - **Unit test:** `src/fusion/_3dof/naive_cf_test.rs (update_mag_correction_tips_level_through_learned_dip)`
