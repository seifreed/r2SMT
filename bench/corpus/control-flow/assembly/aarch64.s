.text
.global r2smt_aarch64_contract
r2smt_aarch64_contract:
    cmp w0, #0
    csel w0, w1, w2, eq
    add v0.4s, v1.4s, v2.4s
    ret
