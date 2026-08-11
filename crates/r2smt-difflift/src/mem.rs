//! Compile the memory effects of two lowerings away, so the equivalence
//! query the harness emits contains no `LoadMem` / `StoreMem` at all.
//!
//! ## Why not just let the encoder model it
//!
//! Memory was excluded from the comparison entirely until 2026-08-06,
//! and that was not laziness — it is what the shared memory model forces.
//! `r2smt_smt::Encoder` keeps **one** byte-store list per slice, so with
//! both lowerings namespaced into a single query side B's loads walk side
//! A's stores: a false agreement, or a real difference masked. And its
//! initial memory is minted *per load*, so two loads of the same address
//! read independently free bytes unless a syntactic memo happens to
//! catch them — which across two lowerings would fabricate a
//! disagreement on every memory instruction, the one direction this
//! harness must never fail in.
//!
//! Scoping the encoder's store list would fix the first and not the
//! second, and would have to be implemented again in the text renderer
//! or cost the cvc5 / bitwuzla portfolio on exactly the new surface.
//!
//! ## What this does instead
//!
//! Every memory effect becomes ordinary `Assign` statements over
//! `Ite` / `Eq` / `Concat` / `Extract` — all `QF_BV`, all already
//! renderable by every backend. Each side keeps its own byte list, so
//! neither can see the other's writes, and both read one **shared**
//! initial memory built as an `Ite` chain over the finite set of byte
//! addresses either side touches:
//!
//! ```text
//! mv_j = ite(k_j == k_{n-1}, mv_{n-1}, … ite(k_j == k_0, mv_0, mb_j))
//! ```
//!
//! The most recently minted key is the outermost `ite`, so on a repeated
//! address the latest mint wins — but that never changes the byte,
//! because each new key is minted *over the existing chain*, forcing an
//! address equal to an earlier key to take that key's value. Equal
//! addresses are thus *forced* to equal bytes transitively, which makes
//! the relation a total function on the key set regardless of `ite`
//! order. That is precisely the property whose absence fabricates.
//!
//! ## Completeness of the probe set
//!
//! Both sides start from the same initial memory, so outside the union of
//! their write footprints both final memories equal it by construction.
//! If the final memories differ at any byte, that byte is therefore in
//! the union of the writes — so probing every written byte is complete,
//! not a heuristic.
//!
//! ## Two implementation details that are not optional
//!
//! Every intermediate is **materialised as its own named definition**,
//! never inlined into the next expression. Z3 hash-conses, but the text
//! renderer emits characters, and an inlined chain is exponential in the
//! number of keys.
//!
//! And the per-byte fold walks a side's stores **oldest → latest**, so
//! the latest write is the outermost `Ite`. Reversed, the oldest matching
//! write would shadow every later overwrite — the same silent unsoundness
//! `Encoder::encode_load_mem` documents.

use std::collections::BTreeMap;

use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;

/// Prefix for every symbol this module mints. Deliberately carries no
/// `#`, so `final_defs` cannot mistake one for an architectural output.
const MEM_SYMBOL_PREFIX: &str = "__m";

/// Distinct byte addresses tracked across both sides. The initial-memory
/// chain is quadratic in this, so it is bounded well below anything a
/// per-instruction lowering reaches.
const MEM_KEY_CAP: usize = 256;

/// Store bytes tracked per side.
const MEM_STORE_CAP: usize = 128;

/// The memory-free rewrite of both lowerings.
pub(crate) struct MemPlan {
    /// Both sides' statements, in order, with every memory effect
    /// expanded into `Assign`s.
    pub(crate) statements: Vec<IrStmt>,
    /// The free initial-memory bytes, to be appended to the query's
    /// inputs.
    pub(crate) free_bytes: Vec<Var>,
    /// Final-memory probe pairs to add to the differ-condition.
    pub(crate) probes: Vec<(Var, Var)>,
}

/// One side's write history: byte address and the byte written there,
/// in program order.
#[derive(Default)]
struct SideMem {
    bytes: Vec<(Var, Var)>,
}

struct Elim {
    ptr_bits: u16,
    /// Byte addresses seen so far and the shared initial-memory byte at
    /// each, in mint order — the chain every later key folds over.
    keys: Vec<(Var, Var)>,
    out: Vec<IrStmt>,
    free: Vec<Var>,
    counter: u32,
}

/// Rewrite both lowerings without memory statements.
///
/// `None` declines the whole comparison. Every decline path fails toward
/// *missing* a disagreement — notably the caps, which decline rather than
/// havocing: havocing one side would leave its loads free while the
/// other's stayed determined, which fabricates.
pub(crate) fn eliminate(
    side_a: &[IrStmt],
    side_b: &[IrStmt],
    ptr_bits: u16,
    probes_enabled: bool,
) -> Option<MemPlan> {
    let mut elim = Elim {
        ptr_bits,
        keys: Vec::new(),
        out: Vec::new(),
        free: Vec::new(),
        counter: 0,
    };
    let mut mem_a = SideMem::default();
    let mut mem_b = SideMem::default();
    elim.run_side(side_a, &mut mem_a)?;
    elim.run_side(side_b, &mut mem_b)?;

    // A side that models no write at all is not disagreeing about memory,
    // it is silent about it — the same category as `OF` / `PF`. Comparing
    // anyway would report every instruction where one engine models a
    // store and the other does not.
    let compare = probes_enabled && !mem_a.bytes.is_empty() && !mem_b.bytes.is_empty();
    let probes = if compare {
        elim.probe_all(&mem_a, &mem_b)?
    } else {
        Vec::new()
    };

    Some(MemPlan {
        statements: elim.out,
        free_bytes: elim.free,
        probes,
    })
}

impl Elim {
    fn fresh(&mut self, tag: &str, bits: u16) -> Option<Var> {
        let n = self.counter;
        self.counter = self.counter.checked_add(1)?;
        Some(Var::new(format!("{MEM_SYMBOL_PREFIX}{tag}{n}"), bits))
    }

    /// Materialise `src` as its own definition and hand back the name.
    fn define(&mut self, tag: &str, bits: u16, src: Expr) -> Option<Var> {
        let dst = self.fresh(tag, bits)?;
        self.out.push(IrStmt::Assign {
            dst: dst.clone(),
            src,
        });
        Some(dst)
    }

    /// The shared initial-memory byte at `addr`.
    fn mem0(&mut self, addr: &Var) -> Option<Var> {
        if self.keys.len() >= MEM_KEY_CAP {
            return None;
        }
        let fresh = self.fresh("b", 8)?;
        self.free.push(fresh.clone());
        let mut value = Expr::Var(fresh);
        for (prev_addr, prev_value) in &self.keys {
            value = Expr::Ite {
                cond: Box::new(Expr::eq(
                    Expr::Var(addr.clone()),
                    Expr::Var(prev_addr.clone()),
                )),
                then_expr: Box::new(Expr::Var(prev_value.clone())),
                else_expr: Box::new(value),
            };
        }
        let defined = self.define("v", 8, value)?;
        self.keys.push((addr.clone(), defined.clone()));
        Some(defined)
    }

    /// Read one byte of `side`'s memory at `addr`, over the shared
    /// initial byte `base`.
    fn fold_side(&mut self, side: &SideMem, addr: &Var, base: &Var) -> Option<Var> {
        let mut value = Expr::Var(base.clone());
        for (store_addr, store_byte) in &side.bytes {
            value = Expr::Ite {
                cond: Box::new(Expr::eq(
                    Expr::Var(addr.clone()),
                    Expr::Var(store_addr.clone()),
                )),
                then_expr: Box::new(Expr::Var(store_byte.clone())),
                else_expr: Box::new(value),
            };
        }
        self.define("r", 8, value)
    }

    fn byte_address(&mut self, base: &Var, index: u16) -> Option<Var> {
        let bits = self.ptr_bits;
        let addr = Expr::add(
            Expr::Var(base.clone()),
            Expr::konst(u128::from(index), bits),
        );
        self.define("a", bits, addr)
    }

    fn run_side(&mut self, stmts: &[IrStmt], side: &mut SideMem) -> Option<()> {
        for stmt in stmts {
            match stmt {
                IrStmt::StoreMem {
                    address,
                    value,
                    bits,
                } => self.expand_store(side, address, value, *bits)?,
                IrStmt::LoadMem { dst, address, bits } => {
                    self.expand_load(side, dst, address, *bits)?;
                }
                other => self.out.push(other.clone()),
            }
        }
        Some(())
    }

    fn expand_store(
        &mut self,
        side: &mut SideMem,
        address: &Expr,
        value: &Expr,
        bits: u16,
    ) -> Option<()> {
        if bits == 0 {
            return Some(());
        }
        let nbytes = bits.div_ceil(8);
        if side.bytes.len().checked_add(usize::from(nbytes))? > MEM_STORE_CAP {
            return None;
        }
        let ptr = self.ptr_bits;
        let base = self.define("sa", ptr, address.clone())?;
        let held = self.define("sv", bits, value.clone())?;
        for i in 0..nbytes {
            let byte_addr = self.byte_address(&base, i)?;
            let lo = i.checked_mul(8)?;
            let hi = lo.checked_add(7)?.min(bits.checked_sub(1)?);
            let byte = self.define("sb", 8, Expr::extract(Expr::Var(held.clone()), hi, lo))?;
            side.bytes.push((byte_addr, byte));
        }
        Some(())
    }

    fn expand_load(&mut self, side: &SideMem, dst: &Var, address: &Expr, bits: u16) -> Option<()> {
        if bits == 0 {
            return Some(());
        }
        let nbytes = bits.div_ceil(8);
        let ptr = self.ptr_bits;
        let base = self.define("la", ptr, address.clone())?;
        let mut acc: Option<Expr> = None;
        for i in 0..nbytes {
            let byte_addr = self.byte_address(&base, i)?;
            let initial = self.mem0(&byte_addr)?;
            let byte = self.fold_side(side, &byte_addr, &initial)?;
            // Little-endian: byte `i` sits above everything read so far.
            acc = Some(match acc.take() {
                None => Expr::Var(byte),
                Some(prev) => Expr::concat(Expr::Var(byte), prev),
            });
        }
        let loaded = acc?;
        let width = nbytes.checked_mul(8)?;
        self.out.push(IrStmt::Assign {
            dst: dst.clone(),
            src: crate::equiv::coerce(loaded, width, bits),
        });
        Some(())
    }

    /// One probe pair per byte either side wrote, deduplicated by the
    /// address definition so a byte written twice is probed once.
    fn probe_all(&mut self, mem_a: &SideMem, mem_b: &SideMem) -> Option<Vec<(Var, Var)>> {
        let mut seen: BTreeMap<String, Var> = BTreeMap::new();
        for (addr, _) in mem_a.bytes.iter().chain(mem_b.bytes.iter()) {
            seen.entry(addr.name.clone())
                .or_insert_with(|| addr.clone());
        }
        let addrs: Vec<Var> = seen.into_values().collect();
        let mut probes = Vec::with_capacity(addrs.len());
        for addr in &addrs {
            let initial = self.mem0(addr)?;
            let a = self.fold_side(mem_a, addr, &initial)?;
            let b = self.fold_side(mem_b, addr, &initial)?;
            probes.push((a, b));
        }
        Some(probes)
    }
}
