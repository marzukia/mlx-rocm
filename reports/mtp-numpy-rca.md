# an earlier run: numpy RCA of F1 (MTP draft uncorrelated with target) — RESOLVED

Date: 2026-10-27 (guest UTC). Submitter: submitter (worker an earlier run, continuation of an earlier run).
Scope: root-cause F1 (draft rank ~206k-248k of 248320, corr ~-0.02..-0.05) on the
mlx-rocm Qwen3.5-9B port (the build VM, gfx1201, ROCm 10.1), validate the numpy
reference, run the never-run E3/E4 ground truth, and settle the L0 GDN r-divergence.

## Verdict (one line)

F1 is a **weight-file norm-convention bug**, not a kernel or math bug:
`model-mtp.safetensors` stored all 7 MTP RMSNorm weights as `(w-1)` (HF raw)
instead of `w` (HF raw + 1), the convention the 4bit file uses for every trunk
norm and the convention `f1step.py` requires (it overwrites `mtp.*` directly from
the file, bypassing the loader's +1 sanitize). Rebuilding the file with
norms = HF+1 makes the draft D = t2 = 8160 at rank 0 (max 17.47). The separate
7x L0 "GDN divergence" was a bug in the numpy reference itself (outer product
written as elementwise product); the K3 GDN kernels are correct.

## E1-E5 status table

| # | experiment | gate | result | status |
|---|-----------|------|--------|--------|
| E1 | numpy port validation (t1, t2, h1 cos vs mlx) | t1=271, t2=8160, h1 cos>0.99 | t1=271, t2=8160, h1 cos=0.9956 (after GDN fix; was t1=10992) | PASS |
| E2 | mlx trunk reference | t1=271, t2=8160, top8 | t1=271, t2=8160 (established an earlier run, unchanged) | PASS (ref) |
| E3 | numpy MTP fed (h1_ref, e1_ref) -> rank of t2 | draft rank small, corr>0 | draft max=16.54, t2=8160 **rank=0**, corr(draft,target)=0.8014 | PASS |
| E4 | MTP fed mlx (h1_mlx, e1_mlx) -> rank of t2 | draft rank small, corr>0 | draft max=17.42, t2=8160 **rank=0**, corr(draft,target)=0.7967 | PASS |
| E5 | L0 GDN 3-way (K3 kernel vs numpy vs op-loop semantics) + f1step acceptance | kernel == sequential ref; draft D=t2 | K3 r=26.170 vs numpy(fixed) r=26.169 (ratio 1.0000, corr 0.99939); isolated GDN: ys ratio 0.9986 / corr 1.00000, state corr 1.00000, T=1 decode corr 1.00000; f1step: t1=271, t2=8160, **draft D=8160=t2, rank=0, max=17.47** | PASS |

Decision tree from an earlier run (E3 good + E4 bad = mlx trunk corrupt; E3 bad = convention
wrong; both good = mlx MTP path suspect): **E3 and E4 both pass, and f1step (the
real mlx MTP path on the fixed file) gives rank 0** — the mlx trunk hidden state
is good, the MTP math is good, and the mlx MTP module path is good. F1 is fully
explained by the file norm convention.

## Root cause (named, with evidence)

### RC-1 (F1): MTP RMSNorm weights stored as (w-1) in model-mtp.safetensors

- Convention (verified an earlier run on 4 trunk norms): the 4bit file stores trunk RMSNorm
  weights as `w = HF_raw + 1` (e.g. L0 input_layernorm mean 1.0331 = 0.0331+1);
  `load_model` therefore does NOT add +1 (weights pre-sanitized); conv1d is
  [C,K,1] (`has_unsanitized_conv1d=False`).
- `f1step.py` overwrites `mtp.*` parameters **directly from model-mtp.safetensors,
  bypassing the sanitize** — so the file must already hold `w`.
- It did not. an earlier run table (means): input_layernorm loaded 0.0991 (HF raw) vs
  required 1.0991; post_attn 0.2413 vs 1.2413; q_norm 0.7573 vs 1.7573;
  k_norm 0.7445 vs 1.7445. All 7 MTP norm tensors were (w-1).
- Effect: every MTP RMSNorm divides by ~10x too little -> normalized MTP
  activations attenuated ~10x -> draft logits uncorrelated (rank ~206k-248k,
  corr ~-0.02..-0.05). Non-norm MTP keys verified exact vs HF (an earlier run).
- Fix + verification: rebuilt `model-mtp.safetensors` with norms = HF+1 (f32;
  original backed up as `.bak-wminus1`; rebuild script had a shape-flattening
  bug, fixed). `f1step.py` re-run: trunk unchanged (t1=271, t2=8160), draft
  **D=8160=t2, rank=0, max=17.47** (was rank ~206k-248k). E3/E4 (independent
  numpy MTP on HF-f32 weights) agree: rank 0, corr ~0.80.

### RC-2 (work order A/D): numpy reference GDN state update was elementwise, not outer product

- `np-rca-np.py:184` (before fix):
  `state = state + k[:, t, :, None, :] * delta[:, :, None, :]`
  `delta` is [B,H,Dv]; `delta[:, :, None, :]` is [B,H,1,Dv] — the Dv axis lands
  last, so the product with [B,H,1,Dk] is elementwise (Dk=Dv=128), updating only
  the v=0 row of each head. Correct: `delta[:, :, :, None]` ([B,H,Dv,1], outer
  product), matching the op-loop `state + k[..., None, :] * delta[..., None]`.
- Evidence: with the bug, L0 r = 183.335 vs K3 26.170 (7x, per-token ratios
  2.75-11.08, corr 0.30 — not a scale factor); after the fix r = 26.169
  (ratio 1.0000, corr 0.99939). Isolated GDN probe (K3 chunk-scan T=14 and
  recurrent T=1 decode vs sequential f64 recurrence on identical
  q/k/v/g/beta from the model): ys ratio 0.9986, corr 1.00000; state ratio
  0.9988, corr 1.00000; T=1 decode corr 1.00000. The earlier "7x kernel
  discrepancy" was entirely the reference bug.
- The an earlier run K3 verification (fused vs op-loop fallback vs f32 ref, max_abs
  0.0-7.8e-3) and this probe together establish the K3 kernels (chunk-scan,
  kkt_solve, recurrent decode) as correct on gfx1201.
- Side findings (not bugs): (a) the ipz shape mismatch in the an earlier run probe
  (mlx (4096,512) vs np (4096,4096)) is a quantized-layout artifact
  (group_size=8), not a model difference; (b) the mlx op-loop graph cannot run
  in this guest (JIT/comgr fails: `hip/hip_runtime.h` not found — HIP headers
  missing from the guest userland), so the numpy sequential recurrence is the
  op-loop reference; (c) `mx.fast.gated_delta_update` accepts f32 g/beta (the
  wrapper casts to the kernels' bf16 — verified f32 vs bf16 calls give identical
  output); the C++ dispatch only dtype-checks q.

## Fix plan (file:line)

1. **`model-mtp.safetensors` (guest, /models/qwen3-9b-mlx/)** — DONE.
   Rebuilt with the 7 MTP RMSNorm weights = HF_raw + 1 (f32); all other keys
   byte-identical. Original preserved as `model-mtp.safetensors.bak-wminus1`.
   Verified by f1step.py (draft D=t2=8160, rank 0). **Follow-up**: the upstream
   conversion recipe that generates `model-mtp.safetensors` must emit norms in
   the `w` convention (same as trunk norms) — pin this in the conversion PR so
   the file is not regenerated broken.
2. **`np-rca-np.py:184`** — DONE (this commit).
   `delta[:, :, None, :]` -> `delta[:, :, :, None]`. Verified: E1 (t1=271,
   t2=8160, h1 cos 0.9956), E3, E4, E5 all pass.
3. No kernel changes required. `vendor/mlx-rocm-upstream/mlx/backend/rocm/`
   r4d GDN kernels and `gated_delta_update.cpp` dispatch are correct as-is
   (kernels md5-identical between the venv-a build tree and the vendored
   source; only the dispatch differs by the an earlier run contiguity-copy fix, which is
   a no-op on contiguous inputs).

## What this means for qmlx-serve

- The 9B dense Qwen3.5 port on the build VM now has: validated numpy reference (E1),
  validated K3 GDN kernels (E5), validated MTP math (E3/E4), and a working MTP
  draft (f1step: D=t2, rank 0) after the one-file fix.
- Remaining for serve: regenerate/verify `model-mtp.safetensors` in the
  conversion pipeline (item 1 follow-up) and re-run the Phase 1 serve smoke
  (`scripts/smoke-serve.sh`) with the fixed file.

## Assets

- np-rca-np.py (fixed, this commit): numpy trunk+MTP (track/e1/l0/e3/e4),
  bf16 shards, f32 math.
- Guest /tmp: np-rca-*.py probes; np-rca-e1.npz / np-rca-e2.npz /
  np-rca-e3.npz / np-rca-e4.npz (logits, h1, e1, ranks); np-rca-gdn-*.npz
  (GDN isolation).
- E2 mlx reference: t1=271, t2=8160; decode top1 8160 (18.88).
- numpy prompt top8 (validated): [(271,18.95),(198,17.84),(471,16.24),(735,14.60),
  (3437,14.56),(21513,14.05),(220,14.03),(7054,13.93)].
