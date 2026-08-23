.text
.globl r2smt_x86_64_contract
r2smt_x86_64_contract:
    test %edi, %edi
    jz 1f
    pxor %xmm0, %xmm0
    vpxor %ymm1, %ymm1, %ymm1
1:  ret
