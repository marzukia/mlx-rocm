# WORKSTATE — an earlier runc (MTP SPEED LEVER: fast M=2 under the bit-exact rule)

Continuation of burned an earlier run (07:33Z). Orchestrator-verified state (07:35Z) trusted over burned-run notes.

## Timeline
- 07:40Z: recovered burned-run analysis from /srv/<fleet>/work/723/.fleetwork/session.jsonl (170 lines; run died writing the verify script after the wheel built).
- 07:43Z: g3 wheel build started (reused baseline build dir; finished in ~1 min).
- 07:44Z: wheel `mlx-0.32.3.dev20261028+a7d1e7f` (21.5MB) in dist-g3s/.
- 07:47Z: venv-c created (fresh venv + g3 wheel + mlx_lm copied from venv-a); import verifies a7d1e7f.
- 07:50Z: primitive verify PASSED — all 4 geometries BIT-EXACT (maxdiff=0.0):
  - gate/up K=4096 N=12288, down K=12288 N=4096, lm_head K=4096 N=248320, o_proj K=4096 N=4096.
- 07:52Z: m0+m1+m2x3 bench started (flock, fresh process per run, venv-c).

## Checkpoints (pushed)
- guest g3: branch mtp-speed-mlp-lmhead (SHA pending)
- worktree: branch worker/an earlier runc-notes (SHA pending)

## Key findings (analysis)
- Model: 4-bit affine, group_size=64, **has_bias=true** (per-group scales+biases in safetensors).
- **lm_head IS quantized**: U32 [248320, 512], scales/biases [248320, 64] (vocab=248320).
- For M=1..8 (4-bit affine, has_bias, non-batched, transpose, bf16): **wide QMV kernel** (qmv_wide_kernel) is the path (wide_ok gate).
- **Wide kernel per-row is M-independent**: row=blockIdx.y, grid=(N/8, M), each block computes one (row, col-tile); K-loop + per-group FMA + shuffle reduction all M-independent.
- Tiled QMV kernel (qmv_tiled_kernel) also M-independent per-row (m=blockIdx.x).
- **Conclusion: a tiled M=2 CONFIG preserves M=1's per-row FMA order — the config variant is viable (fast + bit-exact).** The M=2 path already runs the M=1 row algorithm.
- Rdna4 tuning: tile_n=24 (54CU/8MB), crossover large=18 → M=2 stays QMV for all MLP+lm_head shapes (M=2 < 18; N>=8192&&K>=2048 → M>=max(8,14)=8, M=2<8).

## an earlier run (M-invariance) intersection
- The sibling line makes M=2..8 rows run the M=1 row algorithm for bit-exactness.
- My analysis: the wide QMV kernel (the M=2 path for this model) is ALREADY M-independent per-row → the M=2 path already runs the M=1 row algorithm. So the an earlier run fix is a no-op at the QMM level for this model (the QMM is already bit-exact).
- The o_proj divergence (an earlier run/an earlier run named op) is NOT at the QMM level (primitive shows o_proj BIT-EXACT). It must be in another part of the m2 pipeline (GDN, attention, or the m2 harness setup).
- Speed track does NOT conflict with the correctness fix: the M=2 path (wide QMV) is fast (m2 ≈ m0) and bit-exact. No mitigation needed.

## Gate (pending bench)
- m0 stream unchanged vs reference (/models/tmp/bench4/m0.out)
- m2 stream BIT-EXACT vs m0
- m2 tok/s >= 65.05
