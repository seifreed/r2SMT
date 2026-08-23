.text
.globl patchable
patchable:
    xorl %eax, %eax
    testl %eax, %eax
    jne .Ldead
    ret
.Ldead:
    movl $1, %eax
    ret
