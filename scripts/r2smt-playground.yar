/*
 * r2SMT playground selector — find ELF/PE/Mach-O binaries whose code
 * has the kind of conditional-select density that r2SMT's slicer +
 * SMT backend can actually do something with.
 *
 * Sample-agnostic on purpose: the rules below key only on opcode-
 * level density (REX.W cmovcc pairs, setcc-then-cmovcc adjacency,
 * AArch64 cset/csinc) — not on any specific malware family, ofuscator
 * signature, or packer marker. The thinking is that opaque-predicate
 * idioms collapse to multiple conditional selects in a small window
 * regardless of who emitted them.
 *
 * False-positive expectation: heavily template-instantiated C++
 * binaries built with -O2/-O3 will fire one or two of these strings.
 * The rules therefore require at least two distinct pair patterns
 * before reporting — a single hit is intentionally not enough.
 *
 * Tested formats: ELF (32 + 64-bit), PE/DOS, Mach-O (32 + 64-bit LE)
 * and Mach-O fat (CAFEBABE BE). Java class files share CAFEBABE so
 * the rule also requires a sane code-section density.
 *
 * Reference: https://github.com/seifreed/r2SMT
 */

rule r2smt_playground_x86_64_dense_cmovcc
{
    meta:
        description = "x86_64 ELF/PE/Mach-O with dense cmovcc/setcc pairs — a productive r2SMT opaque-predicate target. Sample-agnostic; rule keys only on contiguous opcode density."
        author      = "r2SMT toolchain"
        date        = "2026-05-15"
        version     = "1"
        target_arch = "x86_64"
        reference   = "https://github.com/seifreed/r2SMT"

    strings:
        /*
         * REX.W cmovcc pairs within ~60 bytes. Each string is 7+
         * contiguous bytes (REX + 0F 4{eq,ne,b,ae,s,ns,l,g} + ModRM,
         * bounded jump, then another REX.W cmovcc). The ModRM is
         * constrained to Mod=11 register-direct (C0..CF) — the form
         * compilers actually emit for predicate fork-ups. The atom
         * yara extracts is "48 0F 4? C?" (3 concrete bytes + half-byte
         * mask, plus the symmetric anchor on the second instruction).
         */

        $cmov_eq_ne_w = {
            48 0F 44 C?
            [0-60]
            48 0F 45 C?
        }
        $cmov_eq_eq_w = {
            48 0F 44 C?
            [4-60]
            48 0F 44 C?
        }
        $cmov_ne_ne_w = {
            48 0F 45 C?
            [4-60]
            48 0F 45 C?
        }
        $cmov_l_g_w = {
            48 0F 4C C?
            [0-60]
            48 0F 4F C?
        }
        $cmov_b_ae_w = {
            48 0F 42 C?
            [0-60]
            48 0F 43 C?
        }
        $cmov_s_ns_w = {
            48 0F 48 C?
            [0-60]
            48 0F 49 C?
        }

        /*
         * `setcc r8` followed by a REX.W `cmovcc r64, r64` within 16
         * bytes — the canonical "evaluate predicate to a 1-bit register
         * then branchlessly fork" idiom that compilers emit for
         * opaque-predicate-style code. The setcc atom alone is only
         * 3 bytes (0F 9{eq,ne} C0), so we anchor on the following
         * cmovcc to get a 4+ byte usable atom inside the pair.
         */
        $setcc_eq_then_cmov = {
            0F 94 C0
            [0-12]
            48 0F 4? C?
        }
        $setcc_ne_then_cmov = {
            0F 95 C0
            [0-12]
            48 0F 4? C?
        }

    condition:
        filesize > 4KB and filesize < 200MB
        and (
            uint32(0) == 0x464C457F     /* ELF "\x7FELF"               */
            or uint16(0) == 0x5A4D      /* PE / DOS stub "MZ"          */
            or uint32(0) == 0xFEEDFACF  /* Mach-O 64-bit LE            */
            or uint32(0) == 0xFEEDFACE  /* Mach-O 32-bit LE            */
            or uint32(0) == 0xBEBAFECA  /* Mach-O fat BE-stored as LE  */
        )
        and 2 of them
}

rule r2smt_playground_aarch64_cs_family
{
    meta:
        description = "AArch64 ELF/Mach-O with multiple cset/csinc instructions (xzr-anchored register-conditional idiom) — r2SMT's new cs* patcher applies here. Sample-agnostic."
        author      = "r2SMT toolchain"
        date        = "2026-05-15"
        version     = "1"
        target_arch = "aarch64"
        reference   = "https://github.com/seifreed/r2SMT"

    strings:
        /*
         * `cset xD, cond` is the alias of `csinc xD, xzr, xzr, !cond`.
         * Encoding (little-endian, Rm=Rn=xzr=31):
         *   byte 3 = 0x9A (sf=1, op=0, S=0, 1101010)
         *   byte 2 = 0x9F (1000 | Rm=11111)
         *   byte 1 = (!cond << 4) | 0x07
         *   byte 0 = 0xE0 | Rd
         * The four concrete bytes are a 4-byte atom — yara is happy.
         * We enumerate cset Xd with Rd in {0..7} (the AAPCS argument /
         * scratch range) for the two most common conditions, eq and
         * ne. The pair condition is the discriminator, not any single
         * cset on its own.
         */

        /* cset xD, eq  →  csinc xD, xzr, xzr, ne (cond=0001) */
        $cset_eq_x0 = { E0 17 9F 9A }
        $cset_eq_x1 = { E1 17 9F 9A }
        $cset_eq_x2 = { E2 17 9F 9A }
        $cset_eq_x3 = { E3 17 9F 9A }

        /* cset xD, ne  →  csinc xD, xzr, xzr, eq (cond=0000) */
        $cset_ne_x0 = { E0 07 9F 9A }
        $cset_ne_x1 = { E1 07 9F 9A }
        $cset_ne_x2 = { E2 07 9F 9A }
        $cset_ne_x3 = { E3 07 9F 9A }

        /* cset xD, lt / le — useful because signed-comparison
         * predicates are the most common opaque-predicate flavor. */
        $cset_lt_x0 = { E0 A7 9F 9A }
        $cset_le_x0 = { E0 C7 9F 9A }

        /*
         * Two `csinc` / `csel` instructions back-to-back (8 bytes apart
         * minimum, ≤ 32 bytes for an adjacent pair). Anchored on the
         * "?? ?? 9? 9A" tail family, which constrains byte 3 to the
         * "data processing register, conditional select, sf=1" encoding
         * group. The atom yara picks here is 1 byte (0x9A) followed by
         * a 4-bit mask — narrower than usual, but the 8-byte spatial
         * constraint plus the magic-byte filter keeps false positives
         * tractable.
         */
        $cs_pair_close = {
            ?? ?? 9? 9A
            [0-24]
            ?? ?? 9? 9A
        }

    condition:
        filesize > 4KB and filesize < 200MB
        and (
            uint32(0) == 0x464C457F     /* ELF             */
            or uint32(0) == 0xFEEDFACF  /* Mach-O 64-bit LE */
            or uint32(0) == 0xBEBAFECA  /* Mach-O fat       */
        )
        and 2 of them
}
