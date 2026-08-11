//! Type-fence contract: an [`ExploreResult`] can never become a
//! [`SmtResult`]. The exploration engine is unsound, so letting its
//! output masquerade as a solver verdict would be a soundness hole. We
//! prove the conversion does not exist by asserting the code that
//! attempts it fails to compile.

#[test]
fn test_smtresult_cannot_be_constructed_from_explore_result() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile-fail/smtresult_from_explore.rs");
}
