# NOTES-p2-sdpa (internal RCA record)

RCA one-liner: the gfx1201 sdpa_flash_wmma Q-staging hoist wrote both D
chunks into the same DW-wide LDS region; chunk 1 overwrote chunk 0 before
the S MMA read it, so S was computed with the back-half Q only (D=256:
4085/4096 cells wrong, non-uniform per-key corruption). The legacy
in-wheel build (lib 1610c9d0) showed the same class of corruption at
64x64 non-causal bf16 (V=rand maxdiff 0.0806, verdict drop b2453c31).

Evidence: with S/P dump instrumentation, the dumped S of the hoisted build
matched q1@(k0+k1) at maxerr 0.000000 and q0@k0+q1@k1 at maxerr 0.854648;
after the fix the dumped S matches the exact reference at maxerr 1e-6 and
the output maxdiff drops to 8e-4 (bf16 noise).

Fix: stage Q per D-chunk inside the S loop (one DW-wide region, single
syncthreads after the Q+K loads). 8 insertions / 12 deletions in
flash_attention_wmma.hip; no other behavior changes.

Note for reviewers: the per-tile tmax reduces over the full 32-lane wave
(5 shfl_xor steps), so the dumped P carries a row-constant
exp(m_row - m_wave) factor. Online softmax is invariant to any per-row
running max, so O is unaffected (verified: dO 8e-4); the per-row rowsum
reduction stays 4 shfl_xor steps (per-row, 16 lanes).

Check lines (R9700 gfx1201, RDNA4, 32 CU, 64KB LDS; same-session A/B,
base = dcd1c69 lib e87166a4, fixed = 330ecba + this fix lib c512b779):
- test_sdpa_full_head_dim_256: base 3/14 fail (bf16 cache-slice shapes)
  -> fixed 14/14 pass
- sig.py: V=ones maxdiff 0.0000 (gate < 1e-3), V=rand 0.00098 (gate
  < 2e-3), f32 control 0.00051 (unchanged)
- test_fast_sdpa full class: base 28 fails -> fixed 25 fails; the 25 are
  the pre-existing fp16 qsl=1 vector-decode tolerance set (identical on
  both builds; vector path untouched by this change)
- SDPA 4K causal bf16 D=256 GQA 8/2 standalone, 8-call: 195.95 ms base
  -> 180.16 ms fixed (1.09x)
- e2e 4K prefill (harness): 5.967 s base -> 5.931 s fixed
