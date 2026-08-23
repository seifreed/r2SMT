#![no_main]

use libfuzzer_sys::fuzz_target;
use r2smt_patch::PatchManifest;

fuzz_target!(|data: &[u8]| {
    let input = String::from_utf8_lossy(data);
    if let Ok(manifest) = PatchManifest::from_json(&input) {
        let _ = manifest.to_json();
    }
});
