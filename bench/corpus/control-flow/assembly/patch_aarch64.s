.text
.globl _patchable
_patchable:
    mov w0, #0
    cmp w0, w0
    b.ne 1f
    ret
1:
    mov w0, #1
    ret
