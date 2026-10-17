use crate::fusion::_6dof::naive_cf::NaiveCF;
use crate::fusion::{AhrsCorrection, Fusion};
use crate::{rw, Rw};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::thread::JoinHandle;

/// Singleton connection owning the background fusion thread and its shared state.
pub struct Connection {
    /// Latest corrected attitude estimate, shared between the fusion thread and readers.
    pub fusion: Rw<AhrsCorrection>,
    /// Set to stop the background fusion thread.
    pub terminating: Arc<AtomicBool>,
    /// While set, the fusion thread pauses updates to unblock readers of `fusion`.
    pub interrupting: Arc<AtomicBool>, // when interrupting, update is paused, opening the fusion mutex for reading
    /// Handle of the background fusion thread, if started.
    pub thread: Option<JoinHandle<()>>,
}

static CONNECTION: Mutex<Option<Connection>> = Mutex::new(None);

impl Connection {
    const ORDERING: Ordering = Ordering::SeqCst;

    /// Return the singleton connection, creating it on first use.
    ///
    /// When `fusion` is `None`, creation falls back to the default 6-DoF
    /// fusion (attitude plus world-frame linear velocity) over whichever
    /// glasses are connected; when the connection already exists, `fusion`
    /// is ignored.
    fn get_or_new(
        fusion: Option<Box<dyn Fusion>>,
    ) -> crate::Result<MutexGuard<'static, Option<Connection>>> {
        let mut existing = Self::get_locked()?;

        if existing.is_none() {
            let fusion = match fusion {
                Some(fusion) => fusion,
                None => NaiveCF::any()?,
            };
            // let ahrs = AHRS::frd(fusion);
            let ahrs = AhrsCorrection::left_fru_down(fusion);

            *existing = Some(Connection {
                fusion: rw(ahrs),
                terminating: Arc::new(AtomicBool::new(false)),
                interrupting: Arc::new(AtomicBool::new(false)), // thread: None,
                thread: None,
            });
        }

        Ok(existing)
    }

    fn get_locked() -> crate::Result<MutexGuard<'static, Option<Connection>>> {
        CONNECTION
            .lock()
            .map_err(|_| crate::Error::ConcurrencyError)
    }

    /// Clear the terminating and interrupting flags on this connection.
    pub fn _init(&self) -> crate::Result<()> {
        self.terminating.store(false, Self::ORDERING);
        self.interrupting.store(false, Self::ORDERING);

        Ok(())
    }

    fn _start(&mut self) -> crate::Result<()> {
        if self.thread.is_some() {
            return Ok(());
        }

        self._init()?;

        let _fusion = self.fusion.clone();
        let _terminating = self.terminating.clone();
        let _interrupting = self.interrupting.clone();

        let handle = thread::spawn(move || loop {
            if _terminating.load(Self::ORDERING) {
                break;
            }

            if _interrupting.load(Self::ORDERING) {
                // println!("busy, no update")
            } else {
                // a poisoned fusion state means `update` or a reader callback
                // panicked; the failure has already surfaced on its own
                // thread, so exiting here avoids a second panic
                let Ok(mut ff) = _fusion.lock() else {
                    break;
                };

                ff.update();
                // println!("UPDATE!")
            }
        });

        self.thread = Some(handle);

        Ok(())
    }

    /// Lazily create the singleton connection and start its background fusion thread.
    ///
    /// When `fusion` is `None`, the default 6-DoF fusion (attitude plus
    /// world-frame linear velocity) over whichever glasses are connected is
    /// used; when the connection already exists, `fusion` is ignored.
    pub fn start(fusion: Option<Box<dyn Fusion>>) -> crate::Result<()> {
        let mut existing = Self::get_or_new(fusion)?;
        let conn = existing.as_mut().ok_or(crate::Error::ConcurrencyError)?;

        conn._start()
    }

    fn _stop(&mut self) -> crate::Result<()> {
        self.terminating.store(true, Self::ORDERING);

        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| crate::Error::Other("connection thread panicked"))?;
        }

        Ok(())
    }

    /// Stop and remove the singleton connection, joining its fusion thread.
    pub fn stop() -> crate::Result<()> {
        let mut existing = Self::get_locked()?;

        if let Some(mut conn) = existing.take() {
            conn._stop()?;
        }

        Ok(())
    }

    /// Pause the fusion thread, run `f` on the latest [`AhrsCorrection`], and resume updates.
    ///
    /// Returns [`crate::Error::Other`] with "fusion state poisoned" when the
    /// fusion state's mutex is poisoned (its thread or a reader panicked);
    /// the connection must be stopped and started again to recover.
    pub fn read_fusion<T>(f: &dyn Fn(&mut AhrsCorrection) -> T) -> crate::Result<T> {
        let (_fusion, _interrupting) = {
            let existing = Self::get_or_new(None)?;

            let conn = existing.as_ref().ok_or(crate::Error::ConcurrencyError)?;
            (conn.fusion.clone(), conn.interrupting.clone())
        };

        _interrupting.store(true, Self::ORDERING);
        // a poisoned fusion state means `update` or a reader callback
        // panicked; report the failure instead of panicking here, where it
        // would unwind across the C ABI
        let result = match _fusion.lock() {
            Ok(mut ahrs) => Ok(f(&mut ahrs)),
            Err(_) => Err(crate::Error::Other("fusion state poisoned")),
        };

        _interrupting.store(false, Self::ORDERING);
        result
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self._stop();
    }
}

#[cfg(test)]
#[path = "connection_tests.rs"]
mod connection_tests;
