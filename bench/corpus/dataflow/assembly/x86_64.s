.text
.globl update_state
update_state:
    testl %edi, %edi
    je .Leven
    addl %edi, state(%rip)
    movl state(%rip), %eax
    ret
.Leven:
    xorl %edi, state(%rip)
    movl state(%rip), %eax
    ret
