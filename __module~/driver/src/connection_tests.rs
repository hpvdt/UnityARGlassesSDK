use serial_test::serial;

use super::Connection;
use crate::fusion::_3dof::naive_cf::NaiveCF;
use crate::fusion::Fusion;
use crate::SimMotion;

fn sim_fusion() -> Box<dyn Fusion> {
    Box::new(NaiveCF::new(Box::new(SimMotion::new())).unwrap())
}

fn serial() -> String {
    Connection::read_fusion(&|ahrs| ahrs.glasses().serial().unwrap()).unwrap()
}

/// The `start` fusion argument is used on creation, ignored while running,
/// and `None` falls back to the default discovery fusion.
#[test]
#[serial]
fn start_uses_given_fusion_and_default_fallback() {
    Connection::stop().unwrap();

    Connection::start(Some(sim_fusion())).unwrap();
    assert_eq!(serial(), "sim_motion!");

    // While running, a new argument is ignored: the first fusion keeps serving.
    Connection::start(Some(sim_fusion())).unwrap();
    assert_eq!(serial(), "sim_motion!");
    Connection::stop().unwrap();

    // `None` falls back to the default fusion over any connected glasses.
    Connection::start(None).unwrap();
    assert!(!serial().is_empty());
    Connection::stop().unwrap();
}
