#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};

use super::*;
use crate::collector::collect_branches;

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address(addr),
        size,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn one_block_program(insns: Vec<Instruction>) -> Program {
    Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: insns,
                successors: vec![],
            }],
            is_thumb: false,
        }],
    }
}

fn slice_first(program: &Program) -> Slice {
    let candidates = collect_branches(program);
    let cand = candidates.first().expect("at least one branch");
    slice_branch(
        cand,
        &program.functions[0],
        &SliceLimits::default(),
        program.arch,
    )
}

#[test]
fn canonical_opaque_predicate_yields_complete_slice() {
    // The SCC example: ((ecx * ecx) & 1) == 2 — always false.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            3,
            "imul",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1005,
            3,
            "and",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1008,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_100b,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert_eq!(slice.status, SliceStatus::Complete);
    let mnemonics: Vec<_> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(mnemonics, vec!["mov", "imul", "and", "cmp"]);
    assert_eq!(slice.roots, vec!["rcx".to_string()]);
}

#[test]
fn unmodeled_register_writer_truncates_instead_of_being_skipped() {
    // `not eax` is unmodeled (effect kind Other). It rewrites the live `eax`
    // that `cmp eax, 0` reads, so the slice must truncate — silently walking
    // past it would resolve `eax` to the stale `xor` value (0) and fabricate
    // an AlwaysTrue verdict for the `je`.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(0x40_1002, 2, "not", vec![op("eax", OperandKind::Register)]),
        insn(
            0x40_1004,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1007,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(
        reason.contains("not"),
        "reason should name the mnemonic: {reason}"
    );
}

#[test]
fn benign_nop_does_not_over_truncate_the_slice() {
    // `nop` is unmodeled (Other) but architecturally side-effect-free, so the
    // walker must step over it and still complete the slice.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(0x40_1002, 1, "nop", vec![]),
        insn(
            0x40_1003,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1006,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert_eq!(slice.status, SliceStatus::Complete);
}

#[test]
fn constant_propagation_has_no_roots() {
    // mov eax, 1; cmp eax, 1; jne junk → always false (constant).
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            5,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1005,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1008,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert_eq!(slice.status, SliceStatus::Complete);
    assert_eq!(slice.instructions.len(), 2);
    assert!(slice.roots.is_empty(), "roots: {:?}", slice.roots);
}

#[test]
fn xor_zero_idiom_terminates_slice_without_roots() {
    // xor eax,eax; test eax,eax; jnz junk
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1004,
            6,
            "jnz",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert_eq!(slice.status, SliceStatus::Complete);
    let mnemonics: Vec<_> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(mnemonics, vec!["xor", "test"]);
    assert!(slice.roots.is_empty());
}

#[test]
fn call_before_flag_producer_truncates() {
    // call f; cmp eax, 0; je → truncated because the call destroys
    // every assumption we have about eax.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            5,
            "call",
            vec![op("0x402000", OperandKind::Immediate)],
        ),
        insn(
            0x40_1005,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1008,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert!(matches!(slice.status, SliceStatus::Truncated { .. }));
    // The flag producer (`cmp`) is still in the slice; the truncation
    // happens when we walk into the `call`.
    assert_eq!(slice.instructions.len(), 1);
    assert_eq!(slice.instructions[0].mnemonic, "cmp");
}

#[test]
fn memory_load_truncates_when_not_allowed() {
    // mov eax, [rax]; cmp eax, 5; je — `[rax]` is unresolved memory
    // (not a stack slot) so the slicer must still truncate.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            3,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("[rax]", OperandKind::Memory),
            ],
        ),
        insn(
            0x40_1003,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("5", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1006,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    assert!(matches!(slice.status, SliceStatus::Truncated { .. }));
}

#[test]
fn missing_flag_producer_truncates() {
    // mov eax, ebx; jne junk — no cmp/test ever sets ZF.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("ebx", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("flag-defining"));
}

#[test]
fn instruction_limit_truncates() {
    let mut insns: Vec<Instruction> = Vec::new();
    // 50 nops worth of dependency chain.
    for i in 0_u32..50 {
        insns.push(insn(
            0x40_1000 + u64::from(i),
            1,
            "add",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ));
    }
    insns.push(insn(
        0x40_1100,
        3,
        "cmp",
        vec![
            op("eax", OperandKind::Register),
            op("100", OperandKind::Immediate),
        ],
    ));
    insns.push(insn(
        0x40_1103,
        6,
        "jne",
        vec![op("0x401080", OperandKind::Immediate)],
    ));
    let program = one_block_program(insns);
    let cand = collect_branches(&program).into_iter().next().unwrap();
    let limits = SliceLimits {
        max_instructions: 8,
        ..SliceLimits::default()
    };
    let slice = slice_branch(&cand, &program.functions[0], &limits, program.arch);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected truncated");
    };
    assert!(reason.contains("instruction limit"));
    assert!(slice.instructions.len() <= 8);
}

#[test]
fn json_round_trips() {
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1004,
            6,
            "jnz",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let slice = slice_first(&program);
    let json = serde_json::to_string(&slice).unwrap();
    let back: Slice = serde_json::from_str(&json).unwrap();
    assert_eq!(back, slice);
}

// --- Multi-block slicing ---

/// Build a function with two linear blocks: `A` falls through to
/// `B`. The branch lives at the end of `B`.
fn two_block_function(a_insns: Vec<Instruction>, b_insns: Vec<Instruction>) -> Program {
    let b_addr = b_insns.first().map_or(Address(0x40_2000), |i| i.address);
    Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![
                BasicBlock {
                    address: Address(0x40_1000),
                    instructions: a_insns,
                    successors: vec![b_addr],
                },
                BasicBlock {
                    address: b_addr,
                    instructions: b_insns,
                    successors: vec![],
                },
            ],
            is_thumb: false,
        }],
    }
}

fn slice_first_with(program: &Program, limits: &SliceLimits) -> Slice {
    let candidates = collect_branches(program);
    let cand = candidates.first().expect("at least one branch");
    slice_branch(cand, &program.functions[0], limits, program.arch)
}

#[test]
fn multi_block_resolves_definition_in_predecessor() {
    // Block A: `mov ecx, 5`
    // Block B: `imul eax, ecx, ecx ; and eax, 1 ; cmp eax, 2 ; jne junk`
    //   → ((5 * 5) & 1) == 2 → false, but the slicer only needs the
    //     mov+imul+and+cmp chain to see that. With max_blocks=2 the
    //     slicer pulls in the `mov ecx, 5` and rcx drops out of roots.
    let a = vec![insn(
        0x40_1000,
        5,
        "mov",
        vec![
            op("ecx", OperandKind::Register),
            op("5", OperandKind::Immediate),
        ],
    )];
    let b = vec![
        insn(
            0x40_2000,
            2,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        ),
        insn(
            0x40_2002,
            3,
            "imul",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_2005,
            3,
            "and",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_2008,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_200b,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ];
    let program = two_block_function(a, b);
    let limits = SliceLimits {
        max_basic_blocks: 2,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    assert_eq!(slice.status, SliceStatus::Complete);
    let mnemonics: Vec<_> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    // Execution order: A's mov ecx,5 comes first, then B's chain.
    assert_eq!(mnemonics, vec!["mov", "mov", "imul", "and", "cmp"]);
    assert!(
        slice.roots.is_empty(),
        "multi-block walk should drop rcx root, got {:?}",
        slice.roots
    );
}

#[test]
fn single_block_default_keeps_root_when_definition_lives_upstream() {
    // Same fixture as the multi-block test, but with the default
    // limits (max_basic_blocks=1). The slicer stops in B with rcx
    // as an external input.
    let a = vec![insn(
        0x40_1000,
        5,
        "mov",
        vec![
            op("ecx", OperandKind::Register),
            op("5", OperandKind::Immediate),
        ],
    )];
    let b = vec![
        insn(
            0x40_2000,
            3,
            "imul",
            vec![
                op("eax", OperandKind::Register),
                op("ecx", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        ),
        insn(
            0x40_2003,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_2006,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ];
    let program = two_block_function(a, b);
    let slice = slice_first_with(&program, &SliceLimits::default());
    // Default (max_basic_blocks=1) — the slicer stays in block B
    // and treats rcx as external. Same Complete-with-roots
    // semantic as Phase 3.
    assert_eq!(slice.status, SliceStatus::Complete);
    assert_eq!(slice.roots, vec!["rcx".to_string()]);
}

#[test]
fn multi_block_resolves_flag_producer_in_predecessor() {
    // Block A: `cmp eax, 0` (sets ZF).
    // Block B: `je junk` — needs ZF.
    // Without multi-block this truncates with "no flag-defining
    // instruction found". With max_blocks=2 the slicer pulls in
    // the cmp and the slice is Complete.
    let a = vec![insn(
        0x40_1000,
        3,
        "cmp",
        vec![
            op("eax", OperandKind::Register),
            op("0", OperandKind::Immediate),
        ],
    )];
    let b = vec![insn(
        0x40_2000,
        6,
        "je",
        vec![op("0x401080", OperandKind::Immediate)],
    )];
    let program = two_block_function(a, b);
    let limits = SliceLimits {
        max_basic_blocks: 2,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    assert_eq!(slice.status, SliceStatus::Complete);
    let mnemonics: Vec<_> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(mnemonics, vec!["cmp"]);
}

#[test]
fn multi_block_join_truncates_with_phi_reason() {
    // CFG: A and C both fall through to B. B ends with `je junk`.
    // The slicer walks B in reverse, reaches block entry while
    // still needing flag info, sees TWO predecessors (A and C),
    // and truncates with a join reason.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![
                BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        3,
                        "cmp",
                        vec![
                            op("eax", OperandKind::Register),
                            op("0", OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_2000)],
                },
                BasicBlock {
                    address: Address(0x40_1100),
                    instructions: vec![insn(
                        0x40_1100,
                        3,
                        "cmp",
                        vec![
                            op("eax", OperandKind::Register),
                            op("1", OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_2000)],
                },
                BasicBlock {
                    address: Address(0x40_2000),
                    instructions: vec![insn(
                        0x40_2000,
                        6,
                        "je",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![],
                },
            ],
            is_thumb: false,
        }],
    };
    let limits = SliceLimits {
        max_basic_blocks: 4,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("join"), "reason was: {reason}");
    assert!(reason.contains("predecessors"), "reason was: {reason}");
    assert!(reason.contains("flag-defining"), "reason was: {reason}");
}

// --- Phase 6: opt-in sound join → free-input boundary ---

fn join_program() -> Program {
    // A (cmp eax,0) and C (cmp eax,1) both fall through to B
    // (`je junk`). Walking B in reverse hits a 2-predecessor join.
    Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![
                BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        3,
                        "cmp",
                        vec![
                            op("eax", OperandKind::Register),
                            op("0", OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_2000)],
                },
                BasicBlock {
                    address: Address(0x40_1100),
                    instructions: vec![insn(
                        0x40_1100,
                        3,
                        "cmp",
                        vec![
                            op("eax", OperandKind::Register),
                            op("1", OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_2000)],
                },
                BasicBlock {
                    address: Address(0x40_2000),
                    instructions: vec![insn(
                        0x40_2000,
                        6,
                        "je",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![],
                },
            ],
            is_thumb: false,
        }],
    }
}

#[test]
fn test_join_default_off_truncates_byte_identical() {
    let program = join_program();
    let limits = SliceLimits {
        max_basic_blocks: 4,
        ..SliceLimits::default()
    };
    assert!(!limits.allow_join_merge, "default must be off");
    let slice = slice_first_with(&program, &limits);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("cannot Φ-merge"), "reason: {reason}");
    assert!(
        !slice.treat_truncation_as_inputs,
        "default off must not promote join-live to free inputs"
    );
}

#[test]
fn test_join_merge_promotes_live_to_free_inputs_soundly() {
    let program = join_program();
    let limits = SliceLimits {
        max_basic_blocks: 4,
        allow_join_merge: true,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    // Soundness guard: a join is never reported `Complete` — the
    // verdict must stay derivable only via free inputs (widen-only,
    // confidence-downgraded), never claimed as a resolved slice.
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("merged as free inputs"), "reason: {reason}");
    assert!(
        slice.treat_truncation_as_inputs,
        "allow_join_merge must promote the join-live set to free inputs"
    );
}

#[test]
fn test_allow_join_merge_does_not_promote_non_join_truncations() {
    // A `call` truncation in a single block (no join). The
    // join-scoped promotion must not leak to call / memory /
    // unsupported truncations — only the global
    // `unknowns_on_truncation` does that.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            5,
            "call",
            vec![op("0x402000", OperandKind::Immediate)],
        ),
        insn(
            0x40_1005,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let limits = SliceLimits {
        allow_join_merge: true,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("call"), "reason: {reason}");
    assert!(
        !slice.treat_truncation_as_inputs,
        "join-merge must be scoped to joins; call truncation unaffected"
    );
}

#[test]
fn multi_block_budget_exhausted_with_needs_flags_truncates() {
    // Three linear blocks A → B → C; C ends with `je junk`. With
    // max_blocks=2 the slicer can visit only C and B before the
    // budget runs out. Since B does not set ZF, needs_flags is
    // still true — slice is Truncated.
    let a = vec![insn(
        0x40_1000,
        3,
        "cmp",
        vec![
            op("eax", OperandKind::Register),
            op("0", OperandKind::Immediate),
        ],
    )];
    let b = vec![insn(
        0x40_1100,
        2,
        "mov",
        vec![
            op("ebx", OperandKind::Register),
            op("ecx", OperandKind::Register),
        ],
    )];
    let c = vec![insn(
        0x40_1200,
        6,
        "je",
        vec![op("0x401080", OperandKind::Immediate)],
    )];
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![
                BasicBlock {
                    address: Address(0x40_1000),
                    instructions: a,
                    successors: vec![Address(0x40_1100)],
                },
                BasicBlock {
                    address: Address(0x40_1100),
                    instructions: b,
                    successors: vec![Address(0x40_1200)],
                },
                BasicBlock {
                    address: Address(0x40_1200),
                    instructions: c,
                    successors: vec![],
                },
            ],
            is_thumb: false,
        }],
    };
    let limits = SliceLimits {
        max_basic_blocks: 2,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("budget"), "reason was: {reason}");
    assert!(reason.contains("flag-defining"), "reason was: {reason}");
}

#[test]
fn multi_block_cycle_back_to_self_truncates() {
    // Self-loop: block A ends with `je 0x401000` (target is A
    // itself). The branch's basic block is A. Walking backward in
    // A doesn't find ZF (no cmp), so we look at A's predecessors
    // — A is its own predecessor, and visited contains A, so we
    // report a cycle.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: vec![
                    insn(
                        0x40_1000,
                        2,
                        "mov",
                        vec![
                            op("eax", OperandKind::Register),
                            op("ebx", OperandKind::Register),
                        ],
                    ),
                    insn(
                        0x40_1002,
                        6,
                        "je",
                        vec![op("0x401000", OperandKind::Immediate)],
                    ),
                ],
                successors: vec![Address(0x40_1000)],
            }],
            is_thumb: false,
        }],
    };
    let limits = SliceLimits {
        max_basic_blocks: 4,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.contains("cycle"), "reason was: {reason}");
    assert!(reason.contains("flag-defining"), "reason was: {reason}");
}

#[test]
fn multi_block_call_in_predecessor_truncates_inside_walk() {
    // Block A: `call f ; ret_addr_unused`. Block B: `cmp eax, 0 ;
    // je junk`. With multi-block walk enabled and `allow_calls`
    // off, walking back into A hits the `call` and truncates with
    // the normal "call at <addr>" reason — same code path as the
    // single-block walker, just exercised across a block edge.
    let a = vec![insn(
        0x40_1000,
        5,
        "call",
        vec![op("0x402000", OperandKind::Immediate)],
    )];
    let b = vec![
        insn(
            0x40_2000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_2003,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ];
    let program = two_block_function(a, b);
    let limits = SliceLimits {
        max_basic_blocks: 4,
        ..SliceLimits::default()
    };
    let slice = slice_first_with(&program, &limits);
    // The cmp resolves needs_flags in B; the eax live entry then
    // drives us into A, which truncates on the `call`.
    let SliceStatus::Truncated { reason } = &slice.status else {
        panic!("expected Truncated, got {:?}", slice.status);
    };
    assert!(reason.starts_with("call at "), "reason was: {reason}");
}

// --- Bounded simple-diamond Φ-merge ---

/// `H: cmp ecx,0; je THEN` — a full diamond whose two arms each set
/// `eax`, reconverging at `JOIN: cmp eax,7; je analysed`. The
/// taken edge (`ZF==1` ⇒ `ecx==0`) reaches `THEN`.
fn diamond_program(then_imm: &str, else_imm: &str) -> Program {
    Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![
                BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            3,
                            "cmp",
                            vec![
                                op("ecx", OperandKind::Register),
                                op("0", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1003,
                            6,
                            "je",
                            vec![op("0x401100", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![Address(0x40_1100), Address(0x40_1009)],
                },
                BasicBlock {
                    address: Address(0x40_1009),
                    instructions: vec![insn(
                        0x40_1009,
                        5,
                        "mov",
                        vec![
                            op("eax", OperandKind::Register),
                            op(else_imm, OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_1200)],
                },
                BasicBlock {
                    address: Address(0x40_1100),
                    instructions: vec![insn(
                        0x40_1100,
                        5,
                        "mov",
                        vec![
                            op("eax", OperandKind::Register),
                            op(then_imm, OperandKind::Immediate),
                        ],
                    )],
                    successors: vec![Address(0x40_1200)],
                },
                BasicBlock {
                    address: Address(0x40_1200),
                    instructions: vec![
                        insn(
                            0x40_1200,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("7", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1203,
                            6,
                            "je",
                            vec![op("0x401300", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![],
                },
            ],
            is_thumb: false,
        }],
    }
}

fn slice_join(program: &Program, limits: &SliceLimits) -> Slice {
    let cands = collect_branches(program);
    let join = cands
        .iter()
        .find(|c| c.address == Address(0x40_1203))
        .expect("join branch present");
    slice_branch(join, &program.functions[0], limits, program.arch)
}

#[test]
fn test_bounded_diamond_merge_recovered_as_complete() {
    let program = diamond_program("5", "5");
    let limits = SliceLimits {
        max_basic_blocks: 8,
        allow_join_merge: true,
        ..SliceLimits::default()
    };
    let slice = slice_join(&program, &limits);
    assert_eq!(
        slice.status,
        SliceStatus::Complete,
        "a fully-resolved diamond is a sound complete slice"
    );
    assert!(
        !slice.treat_truncation_as_inputs,
        "a recovered diamond is resolved, not promoted to free inputs"
    );
    assert_eq!(slice.merges.len(), 1);
    let merge = &slice.merges[0];
    assert_eq!(
        merge
            .merged
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>(),
        vec!["rax"]
    );
    assert_eq!(merge.head.taken_target, Some(Address(0x40_1100)));
    assert_eq!(merge.taken_arm.len(), 1);
    assert_eq!(merge.fallthrough_arm.len(), 1);
    assert_eq!(
        merge
            .head_instructions
            .iter()
            .map(|i| i.mnemonic.as_str())
            .collect::<Vec<_>>(),
        vec!["cmp"]
    );
    assert!(
        !slice.roots.contains(&"rax".to_string()),
        "merged register is resolved by the Ite, not a root"
    );
}

#[test]
fn test_bounded_diamond_default_off_byte_identical() {
    // With `allow_join_merge` off the join handling is unchanged:
    // `needs_flags` is already satisfied by `cmp eax,7` in the
    // join block, so the pre-existing finaliser returns a
    // `Complete` slice with the still-pending `rax` carried as an
    // *unresolved root* (→ free SSA input → sound, imprecise).
    // The contract this test pins: off recovers no merge and the
    // merged register stays a root exactly as before.
    let program = diamond_program("5", "5");
    let limits = SliceLimits {
        max_basic_blocks: 8,
        ..SliceLimits::default()
    };
    assert!(!limits.allow_join_merge);
    let slice = slice_join(&program, &limits);
    assert_eq!(slice.status, SliceStatus::Complete);
    assert!(
        slice.merges.is_empty(),
        "default off must recover no Φ-merge"
    );
    assert!(
        slice.roots.contains(&"rax".to_string()),
        "off: merged register stays an unresolved root (byte-identical)"
    );
    assert!(!slice.treat_truncation_as_inputs);
}

#[test]
fn test_bounded_diamond_lift_polarity_taken_edge_is_then_branch() {
    use r2smt_ir::expr::Expr;
    use r2smt_ir::stmt::IrStmt;

    // THEN sets eax=0xb (11), ELSE sets eax=0x16 (22). The head
    // `je` is taken (→ THEN) exactly when its condition is true,
    // so the lowered `Ite` must put THEN's value in `then_expr`.
    let program = diamond_program("11", "22");
    let limits = SliceLimits {
        max_basic_blocks: 8,
        allow_join_merge: true,
        ..SliceLimits::default()
    };
    let slice = slice_join(&program, &limits);
    let lifted = crate::lift::lift_slice(&slice, program.arch);
    let (then_expr, else_expr) = lifted
        .statements
        .iter()
        .find_map(|s| match s {
            IrStmt::Assign {
                dst,
                src:
                    Expr::Ite {
                        then_expr,
                        else_expr,
                        ..
                    },
            } if dst.name == "rax" => Some(((**then_expr).clone(), (**else_expr).clone())),
            _ => None,
        })
        .expect("rax := Ite(...) assignment lowered from the merge");
    assert!(
        then_expr.to_string().contains("0xb"),
        "taken (THEN) value must drive then_expr, got: {then_expr}"
    );
    assert!(
        else_expr.to_string().contains("0x16"),
        "fallthrough (ELSE) value must drive else_expr, got: {else_expr}"
    );
}
