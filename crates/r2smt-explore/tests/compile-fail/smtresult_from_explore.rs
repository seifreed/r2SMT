// The exploration engine's result must not be convertible into a solver
// verdict. This file MUST fail to compile — there is no
// `From<ExploreResult> for SmtResult`.

use r2smt_common::SmtResult;
use r2smt_explore::ExploreResult;

fn main() {
    let r = ExploreResult::inconclusive("no witness");
    let _s: SmtResult = SmtResult::from(r);
}
