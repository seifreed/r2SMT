# r2SMT — compatibility aliases for the native core plugin
#
# Load with:  r2 -i ~/.local/share/radare2/plugins/r2smt.r2 <binary>
# The native core_r2smt plugin is auto-loaded and exposes the `r2smt`
# command directly. These aliases retain the pre-0.3 command names.
#
# Once loaded, position the cursor on a conditional branch and run:
#
#     $r2smt-at           # one-line verdict for the branch at the cursor
#     $r2smt-at-v         # verdict + solver-simplified form + evidence
#     $r2smt-at-ctx       # verdict + decompiled pseudocode context
#     $r2smt-at-patch     # verdict, then write a verified patched sibling
#     $r2smt-solve        # classify + Markdown report for this branch
#     $r2smt-ctx          # solve + decompiler pseudocode (r2ghidra/r2dec)
#     $r2smt-sweep        # one-line verdict for EVERY branch in this fn
#     $r2smt-annotate     # apply CCu comments live for the current session
#     $r2smt-patch        # backup + manifest, write a patched sibling
#
# The plugin supplies the current file, seek, and function address without
# passing them through a shell. Run `r2smt?` for the native command help.

$r2smt-at=r2smt at
$r2smt-at-v=r2smt explain
$r2smt-at-ctx=r2smt ctx
$r2smt-at-patch=r2smt patch
$r2smt-solve=r2smt solve
$r2smt-solve-deep=r2smt solve-deep
$r2smt-ctx=r2smt ctx
$r2smt-sweep=r2smt sweep
$r2smt-annotate=r2smt annotate
$r2smt-patch=r2smt patch
$r2smt-patch-dry=r2smt patch-dry
$r2smt-rollback=r2smt rollback
