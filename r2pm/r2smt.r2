# r2SMT — radare2 macros
#
# Load with:  r2 -i ~/.local/share/radare2/plugins/r2smt.r2 <binary>
#
# Once loaded, position the cursor on a conditional branch and run:
#
#     $r2smt-at           # one-line verdict for the branch at the cursor
#     $r2smt-at-v         # verdict + solver-simplified form + evidence
#     $r2smt-at-ctx       # verdict + decompiled pseudocode context
#     $r2smt-at-patch     # verdict, then patch + refresh the disasm view
#     $r2smt-solve        # classify + Markdown report for this branch
#     $r2smt-ctx          # solve + decompiler pseudocode (r2ghidra/r2dec)
#     $r2smt-sweep        # one-line verdict for EVERY branch in this fn
#     $r2smt-annotate     # apply CCu comments live for the current session
#     $r2smt-patch        # backup + manifest, apply patch, refresh view
#
# These macros shell out to the `r2smt` CLI installed via
# `r2pm -ci r2smt`. They pass the current binary (`${R2_FILE}`) and the
# cursor address (`$$`); `$FB` is the begin address of the current
# function (used by the sweep). Output is the subcommand's stdout; r2
# prints it inline. The `-patch` / `-rollback` macros additionally
# reopen the file (`oo`) and reprint disasm so the patched bytes show
# up in the live view immediately — the analog of IDA's post-patch
# reanalysis.

$r2smt-at=!r2smt at "${R2_FILE}" $$
$r2smt-at-v=!r2smt at "${R2_FILE}" $$ --explain
$r2smt-at-ctx=!r2smt at "${R2_FILE}" $$ --with-decompiler
$r2smt-at-patch=!r2smt at "${R2_FILE}" $$ --patch; oo; pd 8
$r2smt-solve=!r2smt solve "${R2_FILE}" --at $$ --include-suspicious
$r2smt-solve-deep=!r2smt --deep-analysis solve "${R2_FILE}" --at $$ --include-suspicious
$r2smt-ctx=!r2smt solve "${R2_FILE}" --at $$ --include-suspicious --with-decompiler
$r2smt-sweep=!r2smt solve "${R2_FILE}" --function $FB --include-suspicious
$r2smt-annotate=!r2smt annotate "${R2_FILE}" --at $$
$r2smt-patch=!r2smt patch "${R2_FILE}" --at $$ --apply; oo; pd 8
$r2smt-patch-dry=!r2smt patch "${R2_FILE}" --at $$
$r2smt-rollback=!r2smt patch "${R2_FILE}" --rollback; oo; pd 8
