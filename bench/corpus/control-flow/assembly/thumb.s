.syntax unified
.thumb
.text
.global r2smt_thumb_contract
r2smt_thumb_contract:
    cmp r0, #0
    ite eq
    moveq r0, #1
    movne r0, #2
    bx lr
