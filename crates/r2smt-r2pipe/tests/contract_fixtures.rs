//! Frozen, sanitized radare2 response contracts.

#![allow(clippy::unwrap_used)]

use r2smt_r2pipe::parse::{
    parse_aoj, parse_executable_ranges, parse_function_blocks, parse_function_list, parse_info,
    parse_locals, split_pdgsd_by_instruction,
};

const ROOT: &str = "contracts/r2-6.2.0";

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!("contracts/r2-6.2.0/", $name))
    };
}

#[test]
fn sanitized_r2_6_2_contracts_parse() {
    assert_eq!(ROOT, "contracts/r2-6.2.0");
    assert_eq!(parse_info(fixture!("ij.json")).unwrap().bits, 64);
    assert_eq!(parse_function_list(fixture!("aflj.json")).unwrap().len(), 1);
    assert_eq!(
        parse_function_blocks(fixture!("agfj.json")).unwrap().blocks[0]
            .instructions
            .len(),
        2
    );
    assert_eq!(parse_aoj(fixture!("aoj.json")).unwrap().len(), 1);
    assert_eq!(parse_aoj(fixture!("pdj.json")).unwrap().len(), 1);
    assert_eq!(split_pdgsd_by_instruction(fixture!("pdgsd.txt")).len(), 1);
    assert!(
        parse_locals(fixture!("afvj.json"))
            .unwrap()
            .stack_slots
            .is_empty()
    );
    assert_eq!(
        parse_executable_ranges(fixture!("iSj.json")).unwrap().len(),
        1
    );
}
