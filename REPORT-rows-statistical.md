# REPORT-rows-statistical.md — QMV-rows under the STATISTICAL bar (t1307)

Branch `worker/rows-statistical` off `worker/qmv-rows-bench` @2d65738 (the
t1297 verdict: ws[4] rows kernel 3.14x faster, 1–2 ULP off the per-row
reference, K=3 diverges at token 172 — "do not serve" **under the
bit-exact bar**). Per the 11-02 op order this branch ships under the
**statistical bar** (MTPLX: distributionally identical, matched within the
plain path's own noise — NOT token-identical). The bit-exact track
(t1305) is a separate in-flight job; this branch does not touch it.

Full-stack wheel = 81-build vendor delta + gdn-fused-step-tape (conv-row-fix
+ state-None guard) + verify-attention-s4 (`MLX_ROCM_SDPA_S4VA=1`) +
qmv-rows (ws[4], `MLX_ROCM_QMV_ROWS=1` gate, **default off**). A3 = the
first three with rows OFF = the t1292 69.48 tok/s reference config.

- [ ] wheel built + md5-pinned
- [ ] engagement evidence
- [ ] statistical gates (acceptance parity / distribution parity / first-divergence data)
- [ ] bench table + position vs the 100+ target
- [ ] verdict
