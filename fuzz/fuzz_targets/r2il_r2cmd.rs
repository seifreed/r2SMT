#![no_main]

use libfuzzer_sys::fuzz_target;
use r2smt_common::Arch;

fuzz_target!(|data: &[u8]| {
    let input = String::from_utf8_lossy(data);
    let _ = r2smt_r2il::parse_r2cmd(&input, Arch::X86_64);
});
