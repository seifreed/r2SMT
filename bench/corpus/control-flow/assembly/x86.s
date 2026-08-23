.text
.globl r2smt_x86_contract
r2smt_x86_contract:
    movb %al, %ah
    testb %ah, %ah
    jz 1f
    fld1
    faddp
1:  ret
