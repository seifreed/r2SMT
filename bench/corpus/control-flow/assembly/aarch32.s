.syntax unified
.text
.global r2smt_aarch32_contract
r2smt_aarch32_contract:
    cmp r0, #0
    addeq r0, r0, #1
    vadd.f32 s0, s1, s2
    bx lr
