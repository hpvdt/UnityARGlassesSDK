mod naive_cf;

#[cfg(test)]
mod naive_cf_test;

use crate::any_glasses_or_dummy;
use crate::fusion::Fusion;

impl dyn Fusion {
    /// Create a 6-DoF fusion instance (attitude plus world-frame linear
    /// velocity) over whichever glasses are connected, or the SimMotion
    /// device. The 6-DoF counterpart of [`crate::fusion::Fusion::any_cf`]
    /// via `<dyn Fusion>::any_cf`.
    pub fn any_cf_6dof() -> crate::Result<Box<dyn Fusion>> {
        let glasses = any_glasses_or_dummy()?;
        Ok(Box::new(naive_cf::NaiveCF::new(glasses)?))
    }
}
