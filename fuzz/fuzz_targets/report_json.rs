#![no_main]

use libfuzzer_sys::fuzz_target;
use r2smt_report::Report;

fuzz_target!(|data: &[u8]| {
    if let Ok(report) = serde_json::from_slice::<Report>(data) {
        let _ = report.render_json();
    }
});
