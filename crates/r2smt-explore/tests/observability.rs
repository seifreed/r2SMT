//! Process-isolated observability contract for the exploration watchdog.
#![allow(clippy::unwrap_used)]

use std::process::Command;
use std::sync::{Arc, Mutex};

use r2smt_explore::{WaitOutcome, spawn_and_wait};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'writer> tracing_subscriber::fmt::MakeWriter<'writer> for Capture {
    type Writer = Self;

    fn make_writer(&'writer self) -> Self::Writer {
        self.clone()
    }
}

#[test]
fn test_watchdog_timeout_emits_budget_event() {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(capture.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        assert_eq!(spawn_and_wait(cmd, 50).unwrap(), WaitOutcome::TimedOut);
    });
    let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
    assert!(output.contains("explore.worker_timed_out"));
}
