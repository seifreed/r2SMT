#![allow(clippy::unwrap_used)]

use r2smt_common::Arch;

use super::*;

const IJ_FIXTURE: &str = r#"{
        "core": {},
        "bin": {"arch": "x86", "bits": 64, "baddr": "0x100000000"}
    }"#;

const IEJ_FIXTURE: &str = r#"[
        {"vaddr": 4198720, "paddr": 4096, "haddr": 64, "type": "program"}
    ]"#;

const AFLJ_FIXTURE: &str = r#"[
        {"offset": 4198720, "name": "sym.main", "size": 24},
        {"offset": 4198912, "name": "sym.helper", "size": 16}
    ]"#;

const AGFJ_FIXTURE: &str = r#"[
        {
            "name": "sym.main",
            "offset": 4198720,
            "blocks": [
                {
                    "offset": 4198720,
                    "jump": 4198736,
                    "fail": 4198740,
                    "ops": [
                        {
                            "offset": 4198720,
                            "size": 2,
                            "bytes": "31c0",
                            "opcode": "xor eax, eax",
                            "esil": "0,eax,="
                        },
                        {
                            "offset": 4198722,
                            "size": 5,
                            "bytes": "b810000000",
                            "opcode": "mov eax, 0x10"
                        }
                    ]
                }
            ]
        }
    ]"#;

#[test]
fn parse_info_extracts_arch_and_bits() {
    let info = parse_info(IJ_FIXTURE).unwrap();
    assert_eq!(info.arch, Arch::X86_64);
    assert_eq!(info.bits, 64);
    assert_eq!(info.entry, None);
}

#[test]
fn parse_info_rejects_unsupported_arch() {
    let bad = r#"{"bin": {"arch": "mips", "bits": 32}}"#;
    assert!(parse_info(bad).is_err());
    let bad_bits = r#"{"bin": {"arch": "arm", "bits": 16}}"#;
    assert!(parse_info(bad_bits).is_err());
}

#[test]
fn parse_info_accepts_arm_32_and_aarch64() {
    let arm32 = r#"{"bin": {"arch": "arm", "bits": 32}}"#;
    let info = parse_info(arm32).unwrap();
    assert_eq!(info.arch, Arch::Arm);
    assert_eq!(info.bits, 32);
    let aarch64 = r#"{"bin": {"arch": "arm", "bits": 64}}"#;
    let info = parse_info(aarch64).unwrap();
    assert_eq!(info.arch, Arch::Aarch64);
    assert_eq!(info.bits, 64);
}

#[test]
fn parse_entry_returns_first_vaddr() {
    let entry = parse_entry(IEJ_FIXTURE).unwrap();
    assert_eq!(entry, Some(Address(4_198_720)));
}

#[test]
fn parse_entry_empty_list_returns_none() {
    let entry = parse_entry("[]").unwrap();
    assert_eq!(entry, None);
}

#[test]
fn parse_function_list_returns_two_functions() {
    let funcs = parse_function_list(AFLJ_FIXTURE).unwrap();
    assert_eq!(funcs.len(), 2);
    assert_eq!(funcs[0].address, Address(4_198_720));
    assert_eq!(funcs[0].name, "sym.main");
    assert_eq!(funcs[1].name, "sym.helper");
}

#[test]
fn parse_function_blocks_recovers_instructions_and_successors() {
    let func = parse_function_blocks(AGFJ_FIXTURE).unwrap();
    assert_eq!(func.address, Address(4_198_720));
    assert_eq!(func.name.as_deref(), Some("sym.main"));
    assert_eq!(func.blocks.len(), 1);
    let block = &func.blocks[0];
    assert_eq!(
        block.successors,
        vec![Address(4_198_736), Address(4_198_740)]
    );
    assert_eq!(block.instructions.len(), 2);
    assert_eq!(block.instructions[0].mnemonic, "xor");
    assert_eq!(block.instructions[0].bytes, vec![0x31, 0xc0]);
    assert_eq!(block.instructions[1].mnemonic, "mov");
    assert_eq!(
        block.instructions[1].bytes,
        vec![0xb8, 0x10, 0x00, 0x00, 0x00]
    );
    assert_eq!(block.instructions[0].operands.len(), 2);
    assert_eq!(
        block.instructions[0].operands[0].kind,
        OperandKind::Register
    );
    assert_eq!(
        block.instructions[1].operands[1].kind,
        OperandKind::Immediate
    );
}

#[test]
fn parse_function_blocks_rejects_empty_array() {
    // Strict wrapper still treats "no function here" as an error
    // for single-function callers (load_function / block-at).
    assert!(parse_function_blocks("[]").is_err());
}

const ISJ_FIXTURE: &str = r#"[
        {"name":".text","vaddr":4096,"vsize":256,"perm":"-r-x"},
        {"name":".plt","vaddr":4352,"vsize":64,"perm":"-r-x"},
        {"name":".rodata","vaddr":8192,"vsize":512,"perm":"-r--"},
        {"name":".data","vaddr":12288,"vsize":128,"perm":"-rw-"},
        {"name":".bss","vaddr":16384,"vsize":0,"perm":"-rwx"}
    ]"#;

#[test]
fn parse_executable_ranges_extracts_x_perm_sections() {
    let r = parse_executable_ranges(ISJ_FIXTURE).unwrap();
    // .text [4096,4352) and .plt [4352,4416); .bss has vsize 0 so
    // it maps no bytes and is skipped despite the x permission.
    assert_eq!(r, vec![(4096, 4352), (4352, 4416)]);
}

#[test]
fn parse_executable_ranges_excludes_non_exec_sections() {
    let r = parse_executable_ranges(ISJ_FIXTURE).unwrap();
    // .rodata (8192) and .data (12288) must not appear.
    assert!(!address_in_ranges(&r, 8192));
    assert!(!address_in_ranges(&r, 12288));
}

#[test]
fn parse_executable_ranges_empty_when_no_sections() {
    // Stripped binary: r2 reports an empty section array. Callers
    // treat this as "do not filter", never "filter everything".
    assert!(parse_executable_ranges("[]").unwrap().is_empty());
}

#[test]
fn parse_executable_ranges_errors_on_malformed_json() {
    assert!(parse_executable_ranges("{ not json").is_err());
}

#[test]
fn address_in_ranges_is_half_open() {
    let ranges = vec![(0x1000u64, 0x1100u64)];
    assert!(!address_in_ranges(&ranges, 0x0FFF));
    assert!(address_in_ranges(&ranges, 0x1000)); // start inclusive
    assert!(address_in_ranges(&ranges, 0x10FF));
    assert!(!address_in_ranges(&ranges, 0x1100)); // end exclusive
    assert!(!address_in_ranges(&[], 0x1000)); // empty => never inside
}

fn block_at(addr: u64) -> BasicBlock {
    BasicBlock {
        address: Address(addr),
        instructions: Vec::new(),
        successors: Vec::new(),
    }
}

fn func_with_blocks(addr: u64, block_addrs: &[u64]) -> Function {
    Function {
        address: Address(addr),
        name: Some("sym.test".into()),
        blocks: block_addrs.iter().copied().map(block_at).collect(),
        is_thumb: false,
    }
}

#[test]
fn retain_executable_blocks_drops_data_blocks_keeps_text() {
    // .text [0x1000,0x2000). One real block at 0x1000, one block
    // r2 over-extended into a string table at 0x9000 (data).
    let ranges = vec![(0x1000u64, 0x2000u64)];
    let mut f = func_with_blocks(0x1000, &[0x1000, 0x9000, 0x1100]);
    let dropped = retain_executable_blocks(&mut f, &ranges);
    assert_eq!(dropped, 1);
    let kept: Vec<u64> = f.blocks.iter().map(|b| b.address.get()).collect();
    assert_eq!(kept, vec![0x1000, 0x1100]);
}

#[test]
fn retain_executable_blocks_empties_a_fully_data_function() {
    // Every block r2 attributed to this "function" is in data —
    // the whole thing is an analysis artefact. The caller skips
    // functions left with zero blocks.
    let ranges = vec![(0x1000u64, 0x2000u64)];
    let mut f = func_with_blocks(0x0014_0710, &[0x0005_5ebc, 0x0005_6018]);
    let dropped = retain_executable_blocks(&mut f, &ranges);
    assert_eq!(dropped, 2);
    assert!(f.blocks.is_empty());
}

#[test]
fn parse_function_blocks_opt_returns_none_for_empty_array() {
    // r2 returns `[]` for an address with no decoded CFG (import
    // thunk / data / placeholder). The list-walking caller must be
    // able to skip it rather than abort the whole program load.
    assert_eq!(parse_function_blocks_opt("[]").unwrap(), None);
}

#[test]
fn parse_function_blocks_opt_returns_some_for_real_function() {
    let func = parse_function_blocks_opt(AGFJ_MODERN).unwrap();
    assert!(func.is_some());
}

#[test]
fn parse_function_blocks_opt_still_errors_on_malformed_json() {
    // A genuinely corrupt response is a contract violation and
    // must NOT be silently swallowed as "no function here".
    assert!(parse_function_blocks_opt("{not json").is_err());
    assert!(parse_function_blocks_opt("[{\"addr\": \"oops\"}]").is_err());
}

#[test]
fn decode_hex_bytes_rejects_odd_length() {
    assert!(decode_hex_bytes("abc").is_err());
}

#[test]
fn classify_operand_known_cases() {
    assert_eq!(classify_operand("eax"), OperandKind::Register);
    assert_eq!(classify_operand("0x10"), OperandKind::Immediate);
    assert_eq!(classify_operand("[rbp - 4]"), OperandKind::Memory);
    assert_eq!(classify_operand("dword ptr [eax]"), OperandKind::Memory);
    // AArch64 / AArch32 immediates carry a `#` prefix.
    assert_eq!(classify_operand("#0x10"), OperandKind::Immediate);
    assert_eq!(classify_operand("#-1"), OperandKind::Immediate);
    assert_eq!(classify_operand("#42"), OperandKind::Immediate);
    // AArch64 register tokens stay register-classified.
    assert_eq!(classify_operand("x0"), OperandKind::Register);
    assert_eq!(classify_operand("w0"), OperandKind::Register);
}

// The radare2 schema migrated from `offset` to `addr` between r2 5.x
// and 6.x. The parsers use `#[serde(alias = "offset")]` so both
// schemas keep working; these tests pin that behaviour.

const AFLJ_MODERN: &str = r#"[
        {"addr": 4198720, "name": "sym.main", "size": 24}
    ]"#;

const AGFJ_MODERN: &str = r#"[
        {
            "name": "sym.main",
            "addr": 4198720,
            "blocks": [
                {
                    "addr": 4198720,
                    "jump": 4198736,
                    "fail": 4198740,
                    "ops": [
                        {
                            "addr": 4198720,
                            "size": 2,
                            "bytes": "31c0",
                            "opcode": "xor eax, eax"
                        }
                    ]
                }
            ]
        }
    ]"#;

#[test]
fn parse_function_list_accepts_modern_addr_schema() {
    let funcs = parse_function_list(AFLJ_MODERN).unwrap();
    assert_eq!(funcs.len(), 1);
    assert_eq!(funcs[0].address, Address(4_198_720));
}

#[test]
fn parse_function_blocks_accepts_modern_addr_schema() {
    let func = parse_function_blocks(AGFJ_MODERN).unwrap();
    assert_eq!(func.address, Address(4_198_720));
    assert_eq!(func.blocks[0].address, Address(4_198_720));
    assert_eq!(func.blocks[0].instructions[0].address, Address(4_198_720));
}

#[test]
fn parse_function_blocks_marks_thumb_when_bits_16() {
    let json = r#"[
            {
                "addr": 32774,
                "name": "sym.thumb_fn",
                "bits": 16,
                "blocks": [
                    {
                        "addr": 32774,
                        "ops": [
                            {
                                "addr": 32774,
                                "size": 2,
                                "bytes": "00bf",
                                "opcode": "nop"
                            }
                        ]
                    }
                ]
            }
        ]"#;
    let func = parse_function_blocks(json).unwrap();
    assert!(func.is_thumb);
    assert!(func.blocks[0].instructions[0].is_thumb);
}

#[test]
fn parse_function_blocks_defaults_to_arm_mode_when_bits_absent() {
    // Most agfj responses omit `bits`. The parser must default to
    // ARM mode (is_thumb == false) so existing fixtures keep
    // working unchanged.
    let func = parse_function_blocks(AGFJ_MODERN).unwrap();
    assert!(!func.is_thumb);
    assert!(!func.blocks[0].instructions[0].is_thumb);
}

// ----- aoj / axtj / afvj / fdj ----------------------------------

#[test]
fn parse_aoj_classifies_branch_type() {
    let json = r#"[
            {"addr": 4198720, "size": 6, "bytes": "0f8500000000",
             "opcode": "jne 0x401080", "type": "cjmp"},
            {"addr": 4198726, "size": 1, "bytes": "c3",
             "opcode": "ret", "type": "ret"},
            {"addr": 4198727, "size": 5, "bytes": "e900000000",
             "opcode": "jmp 0x401100", "type": "jmp"},
            {"addr": 4198732, "size": 3, "bytes": "488d05",
             "opcode": "mov rax, 1", "type": "mov"}
        ]"#;
    let parsed = parse_aoj(json).unwrap();
    assert_eq!(parsed[0].flow, InsnFlow::ConditionalBranch);
    assert_eq!(parsed[1].flow, InsnFlow::Return);
    assert_eq!(parsed[2].flow, InsnFlow::UnconditionalJump);
    assert_eq!(parsed[3].flow, InsnFlow::Linear);
    assert_eq!(parsed[0].mnemonic, "jne");
    assert_eq!(parsed[1].bytes, vec![0xC3]);
}

#[test]
fn parse_aoj_falls_back_to_mnemonic_when_type_missing() {
    let json = r#"[{"addr": 100, "size": 2, "bytes": "74fe", "opcode": "je 0x60"}]"#;
    let parsed = parse_aoj(json).unwrap();
    assert_eq!(parsed[0].flow, InsnFlow::ConditionalBranch);
}

#[test]
fn insn_flow_ends_block_only_for_terminators() {
    assert!(InsnFlow::ConditionalBranch.ends_block());
    assert!(InsnFlow::UnconditionalJump.ends_block());
    assert!(InsnFlow::Return.ends_block());
    assert!(!InsnFlow::Linear.ends_block());
    assert!(!InsnFlow::Call.ends_block());
}

#[test]
fn parse_xrefs_returns_inbound_refs() {
    let json = r#"[
            {"from": 4198704, "type": "code", "opcode": "call sym.main"},
            {"from": 4198800, "type": "data"}
        ]"#;
    let xrefs = parse_xrefs(json).unwrap();
    assert_eq!(xrefs.len(), 2);
    assert_eq!(xrefs[0].from, Address(4_198_704));
    assert_eq!(xrefs[0].kind.as_deref(), Some("code"));
    assert_eq!(xrefs[1].kind.as_deref(), Some("data"));
}

#[test]
fn parse_xrefs_empty_array_returns_empty() {
    let xrefs = parse_xrefs("[]").unwrap();
    assert!(xrefs.is_empty());
}

#[test]
fn parse_locals_returns_local_names_keyed_by_stack_slot() {
    let json = r#"{
            "bp": [
                {"name": "var_4h", "kind": "v", "type": "int",
                 "ref": {"base": "rbp", "offset": -4}},
                {"name": "arg_8h", "kind": "a", "type": "int",
                 "ref": {"base": "rbp", "offset": 8}}
            ],
            "sp": [
                {"name": "local_10h", "kind": "v", "type": "int",
                 "ref": {"base": "rsp", "offset": 16}}
            ]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert_eq!(locals.stack_slots.len(), 3);
    assert!(locals.registers.is_empty());
    assert!(
        locals
            .stack_slots
            .iter()
            .any(|l| l.name == "var_4h" && l.stack_slot == "stk_rbp_-4")
    );
    assert!(
        locals
            .stack_slots
            .iter()
            .any(|l| l.name == "arg_8h" && l.stack_slot == "stk_rbp_8")
    );
    assert!(
        locals
            .stack_slots
            .iter()
            .any(|l| l.name == "local_10h" && l.stack_slot == "stk_rsp_16")
    );
}

#[test]
fn parse_locals_skips_unsupported_bases() {
    let json = r#"{
            "bp": [{"name": "weird", "kind": "v",
                    "ref": {"base": "rax", "offset": 0}}]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert!(locals.stack_slots.is_empty());
    assert!(locals.registers.is_empty());
}

#[test]
fn parse_locals_handles_empty_object() {
    let locals = parse_locals("{}").unwrap();
    assert!(locals.stack_slots.is_empty());
    assert!(locals.registers.is_empty());
}

#[test]
fn parse_locals_surfaces_register_renames_with_string_ref() {
    // r2 6.x emits register-typed locals as a bare string in
    // `ref`. The parser must surface them under
    // `Locals::registers` without touching `stack_slots`.
    let json = r#"{
            "bp": [],
            "sp": [],
            "reg": [
                {"name": "arg1", "kind": "r", "type": "int", "ref": "rdi"},
                {"name": "userInput", "kind": "r", "type": "char*", "ref": "rsi"}
            ]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert!(locals.stack_slots.is_empty());
    assert_eq!(locals.registers.len(), 2);
    assert!(
        locals
            .registers
            .iter()
            .any(|r| r.name == "arg1" && r.register == "rdi")
    );
    assert!(
        locals
            .registers
            .iter()
            .any(|r| r.name == "userInput" && r.register == "rsi")
    );
}

#[test]
fn parse_locals_accepts_register_ref_in_object_form() {
    // Older r2 builds emitted register entries as
    // `{"base": "rdi", "offset": 0}` instead of a bare string.
    // The parser tolerates this shape too.
    let json = r#"{
            "reg": [
                {"name": "arg1", "kind": "r",
                 "ref": {"base": "rdi", "offset": 0}}
            ]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert_eq!(locals.registers.len(), 1);
    assert_eq!(locals.registers[0].name, "arg1");
    assert_eq!(locals.registers[0].register, "rdi");
}

#[test]
fn parse_locals_normalises_register_name_case() {
    // r2 occasionally emits register names in mixed case
    // (`"RDI"`). The lifter compares against lowercase, so the
    // parser normalises at the boundary.
    let json = r#"{
            "reg": [{"name": "arg1", "ref": "RDI"}]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert_eq!(locals.registers.len(), 1);
    assert_eq!(locals.registers[0].register, "rdi");
}

#[test]
fn parse_locals_drops_register_entries_with_empty_payload() {
    let json = r#"{
            "reg": [
                {"name": "ok", "ref": "rdi"},
                {"name": "missing_ref"},
                {"name": "blank", "ref": ""},
                {"ref": "rsi"}
            ]
        }"#;
    let locals = parse_locals(json).unwrap();
    assert_eq!(locals.registers.len(), 1);
    assert_eq!(locals.registers[0].name, "ok");
}

#[test]
fn parse_flag_returns_name_for_object_form() {
    let json = r#"{"name": "sym.main", "offset": 4198720, "size": 0}"#;
    let flag = parse_flag(json).unwrap();
    assert_eq!(flag.as_deref(), Some("sym.main"));
}

#[test]
fn parse_flag_returns_first_name_for_array_form() {
    let json = r#"[{"name": "sym.first"}, {"name": "sym.second"}]"#;
    let flag = parse_flag(json).unwrap();
    assert_eq!(flag.as_deref(), Some("sym.first"));
}

#[test]
fn parse_flag_returns_none_for_empty_responses() {
    assert_eq!(parse_flag("").unwrap(), None);
    assert_eq!(parse_flag("null").unwrap(), None);
    assert_eq!(parse_flag("[]").unwrap(), None);
    assert_eq!(parse_flag("{}").unwrap(), None);
}

#[test]
fn parse_pdgj_extracts_code_field() {
    let json = r#"{"code": "int main() {\n  return 0;\n}\n", "annotations": []}"#;
    assert_eq!(
        parse_pdgj(json).as_deref(),
        Some("int main() {\n  return 0;\n}")
    );
}

#[test]
fn parse_pdgj_returns_none_when_backend_absent() {
    assert_eq!(parse_pdgj("Cannot find decompiler for current arch"), None);
    assert_eq!(parse_pdgj(""), None);
    assert_eq!(parse_pdgj("{}"), None);
    assert_eq!(parse_pdgj(r#"{"code": "   "}"#), None);
}

#[test]
fn parse_pddj_joins_line_fragments() {
    let json = r#"{"errors":[],"log":[],"lines":[{"str":"int f(void) {"},{"str":"  return 1;"},{"str":"}"}]}"#;
    assert_eq!(
        parse_pddj(json).as_deref(),
        Some("int f(void) {\n  return 1;\n}")
    );
}

#[test]
fn parse_pddj_returns_none_without_lines() {
    assert_eq!(parse_pddj(r#"{"errors":["r2dec failed"]}"#), None);
    assert_eq!(parse_pddj("not json"), None);
    assert_eq!(parse_pddj(r#"{"lines":[]}"#), None);
}

#[test]
fn clean_plain_decompile_filters_sentinels_but_keeps_code() {
    assert_eq!(clean_plain_decompile("Unknown command 'pdg'"), None);
    assert_eq!(clean_plain_decompile("   "), None);
    assert_eq!(
        clean_plain_decompile("undefined4 main(void)\n{\n  return 0;\n}\n").as_deref(),
        Some("undefined4 main(void)\n{\n  return 0;\n}")
    );
}

#[test]
fn split_pdgsd_groups_ops_under_each_instruction() {
    let dump = "\
0x100: sub sp, sp, #0x10
    sp = INT_SUB sp, 0x10
0x104: mul w8, w8, w9
    (unique,0x2ae80,4) = INT_MULT w8, w9
    x8 = INT_ZEXT (unique,0x2ae80,4)";
    let groups = split_pdgsd_by_instruction(dump);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].0, 0x100);
    assert_eq!(
        groups[0].1,
        "0x100: sub sp, sp, #0x10\n    sp = INT_SUB sp, 0x10"
    );
    assert_eq!(groups[1].0, 0x104);
    assert!(groups[1].1.contains("INT_MULT w8, w9"));
    assert!(groups[1].1.contains("INT_ZEXT"));
}

#[test]
fn split_pdgsd_ignores_stray_log_lines() {
    let dump = "WARN: something\n0x100: nop\n    --- nop has no ops\nERROR: noise";
    let groups = split_pdgsd_by_instruction(dump);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0, 0x100);
}
