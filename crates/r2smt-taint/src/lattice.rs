//! The taint join-semilattice: a set of taint sources per value.
//!
//! Bottom is the empty set (untainted). The join is set union, so the
//! lattice is finite (bounded by the number of seeded sources) and a
//! forward pass over SSA reaches its fixed point in one sweep.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// An opaque identifier for a taint source (e.g. "argv[1]", "stdin",
/// "the RC4 key"). The caller assigns meaning; the lattice only unions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceId(pub u32);

impl SourceId {
    /// Construct from a raw index.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// The taint of a single value: the set of sources it may carry. Empty
/// means untainted (the lattice bottom).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaintSet(BTreeSet<SourceId>);

impl TaintSet {
    /// The untainted bottom element.
    #[must_use]
    pub fn untainted() -> Self {
        Self(BTreeSet::new())
    }

    /// A singleton taint carrying exactly `source`.
    #[must_use]
    pub fn source(source: SourceId) -> Self {
        Self(BTreeSet::from([source]))
    }

    /// Whether this value is untainted (bottom).
    #[must_use]
    pub fn is_untainted(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether `source` is among this value's taints.
    #[must_use]
    pub fn contains(&self, source: SourceId) -> bool {
        self.0.contains(&source)
    }

    /// Join `other` into `self` (set union). Returns whether `self`
    /// changed — useful for fixed-point loops.
    pub fn join_from(&mut self, other: &Self) -> bool {
        let before = self.0.len();
        self.0.extend(other.0.iter().copied());
        self.0.len() != before
    }

    /// The join of `self` and `other` as a fresh set.
    #[must_use]
    pub fn joined(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.join_from(other);
        out
    }

    /// Iterate the sources in ascending order.
    pub fn sources(&self) -> impl Iterator<Item = SourceId> + '_ {
        self.0.iter().copied()
    }
}

impl FromIterator<SourceId> for TaintSet {
    fn from_iter<I: IntoIterator<Item = SourceId>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}
