//! Optional, human-readable aliases for the canonical names emitted by
//! the lifter / SSA pass.
//!
//! The IR keeps deterministic canonical names — `stk_rbp_-4` for a
//! stack slot, `rax#2` for an SSA-renamed register — so that downstream
//! passes can pattern-match without parsing English. For human reports
//! those names are noisy: an analyst would rather see `var_4h` (the
//! name radare2 assigned to the slot) or `sym.config_table` (the
//! global flag at that address).
//!
//! [`NameHints`] is the side-channel carrying those aliases. It is
//! produced by the adapter ([`crate::BinaryProvider::name_hints`]) and
//! consumed by the report / pretty-printer layer. The IR itself never
//! reads it, so the substitution is purely cosmetic and cannot make
//! the verdict unsound.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Optional pretty-name overrides keyed by the canonical names that
/// the lifter emits.
///
/// All maps are *additive*: an empty `NameHints` means "no aliases";
/// the canonical name is used everywhere.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NameHints {
    /// Canonical stack-slot name (e.g. `"stk_rbp_-4"`) → analyst-facing
    /// alias (e.g. `"var_4h"` or `"arg_8h"`).
    pub stack_slots: BTreeMap<String, String>,
    /// Absolute address of a global → its symbolic name
    /// (e.g. `0x401050` → `"sym.config_table"`). Surfaced by the
    /// report layer when a memory operand resolves to a constant
    /// address.
    pub globals: BTreeMap<u64, String>,
    /// Canonical register name (e.g. `"rax"`) → analyst-facing alias,
    /// if one is supplied by the adapter. The radare2 adapter
    /// populates this from the `reg` array of `afvj @ fn`: when an
    /// analyst (or the calling-convention pass) names a value that
    /// lives in a register — `arg1` → `rdi`, `userInput` → `rsi`,
    /// `dst` → `x0` — the pretty-printer surfaces that name instead
    /// of the bare register. Functions without register-typed locals
    /// leave the map empty and `register(...)` returns the canonical
    /// name unchanged.
    #[serde(default)]
    pub registers: BTreeMap<String, String>,
}

impl NameHints {
    /// Build an empty hint set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an alias for a stack slot. Last write wins so callers
    /// can rebuild the map without bookkeeping.
    pub fn add_stack_slot(&mut self, canonical: impl Into<String>, alias: impl Into<String>) {
        self.stack_slots.insert(canonical.into(), alias.into());
    }

    /// Record the human name for a global symbol at `address`.
    pub fn add_global(&mut self, address: u64, alias: impl Into<String>) {
        self.globals.insert(address, alias.into());
    }

    /// Record an analyst-facing alias for a canonical register name.
    /// Last write wins.
    pub fn add_register(&mut self, canonical: impl Into<String>, alias: impl Into<String>) {
        self.registers.insert(canonical.into(), alias.into());
    }

    /// Resolve a canonical stack-slot name to its alias. Returns the
    /// canonical name unchanged if no alias is registered.
    #[must_use]
    pub fn stack_slot<'a>(&'a self, canonical: &'a str) -> &'a str {
        self.stack_slots
            .get(canonical)
            .map_or(canonical, String::as_str)
    }

    /// Resolve a canonical register name to its alias. Returns the
    /// canonical name unchanged if no alias is registered (the
    /// expected outcome today; hardware registers have no human
    /// alias).
    #[must_use]
    pub fn register<'a>(&'a self, canonical: &'a str) -> &'a str {
        self.registers
            .get(canonical)
            .map_or(canonical, String::as_str)
    }

    /// Resolve an absolute address to its global name, if any.
    #[must_use]
    pub fn global(&self, address: u64) -> Option<&str> {
        self.globals.get(&address).map(String::as_str)
    }

    /// `true` if no aliases are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stack_slots.is_empty() && self.globals.is_empty() && self.registers.is_empty()
    }

    /// Merge `other` into `self`. Entries in `other` overwrite
    /// existing keys.
    pub fn extend(&mut self, other: NameHints) {
        self.stack_slots.extend(other.stack_slots);
        self.globals.extend(other.globals);
        self.registers.extend(other.registers);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn empty_hint_set_returns_canonical_unchanged() {
        let hints = NameHints::new();
        assert!(hints.is_empty());
        assert_eq!(hints.stack_slot("stk_rbp_-4"), "stk_rbp_-4");
        assert_eq!(hints.global(0x40_1050), None);
        assert_eq!(hints.register("rax"), "rax");
    }

    #[test]
    fn register_alias_lookup_returns_alias_when_present() {
        let mut hints = NameHints::new();
        hints.add_register("rax", "result");
        assert_eq!(hints.register("rax"), "result");
        // Unmapped canonical name passes through.
        assert_eq!(hints.register("rcx"), "rcx");
        assert!(!hints.is_empty());
    }

    #[test]
    fn stack_slot_alias_lookup_returns_alias_when_present() {
        let mut hints = NameHints::new();
        hints.add_stack_slot("stk_rbp_-4", "var_4h");
        assert_eq!(hints.stack_slot("stk_rbp_-4"), "var_4h");
        // Unmapped canonical name passes through.
        assert_eq!(hints.stack_slot("stk_rbp_-8"), "stk_rbp_-8");
        assert!(!hints.is_empty());
    }

    #[test]
    fn global_lookup_returns_name() {
        let mut hints = NameHints::new();
        hints.add_global(0x40_1050, "sym.main");
        assert_eq!(hints.global(0x40_1050), Some("sym.main"));
        assert_eq!(hints.global(0x40_1051), None);
    }

    #[test]
    fn extend_merges_and_overwrites_keys() {
        let mut a = NameHints::new();
        a.add_stack_slot("stk_rbp_-4", "var_4h");
        a.add_global(0x40_1000, "sym.first");
        let mut b = NameHints::new();
        b.add_stack_slot("stk_rbp_-4", "renamed");
        b.add_global(0x40_2000, "sym.second");
        a.extend(b);
        assert_eq!(a.stack_slot("stk_rbp_-4"), "renamed");
        assert_eq!(a.global(0x40_1000), Some("sym.first"));
        assert_eq!(a.global(0x40_2000), Some("sym.second"));
    }

    #[test]
    fn json_round_trip_preserves_maps() {
        let mut hints = NameHints::new();
        hints.add_stack_slot("stk_rbp_-4", "var_4h");
        hints.add_global(0x40_1050, "sym.config");
        hints.add_register("rax", "result");
        let json = serde_json::to_string(&hints).unwrap();
        let back: NameHints = serde_json::from_str(&json).unwrap();
        assert_eq!(back, hints);
    }

    #[test]
    fn legacy_json_without_registers_field_still_deserialises() {
        // Snapshots written by older versions of r2SMT predate the
        // `registers` field. `#[serde(default)]` makes the field
        // backward-compatible: an absent map deserialises as empty.
        let json = r#"{"stack_slots":{"stk_rbp_-4":"var_4h"},"globals":{}}"#;
        let hints: NameHints = serde_json::from_str(json).unwrap();
        assert_eq!(hints.stack_slot("stk_rbp_-4"), "var_4h");
        assert!(hints.registers.is_empty());
    }
}
