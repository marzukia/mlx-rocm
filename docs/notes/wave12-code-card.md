# WAVE-12 CODE CARD — 2.5-bit 27B loader mechanism + path enumeration
# (code reviewer 1/3, zero-GPU)

Date: 2026-11-26 (UTC). Tree: mirror origin/main @ 3e84320. Build-input pin:
vendor.json aed6a25 (2026-11-13); on-box wheel +75ca881 (2026-11-03).
All line refs are origin/main unless prefixed `pin:` or `master:`.

## A. Mechanism verification (the terminal card's five claims)

### (1) gguflib reader + type table — VERIFIED
- Reader wiring: `vendor/mlx-rocm-upstream/mlx/io/CMakeLists.txt:13-14` —
  FetchContent pins antirez/gguf-tools at 8fa6eb65 (full sha
  8fa6eb65236618e28fd7710a0fba565f7faa1848; 2025-01-09, "Merge PR #16
  ftz"; commit exists upstream, verified 2026-11-26). Identical pin in the
  vendor pin: the aed6a25 tree carries the same CMakeLists.txt:14 line.
- f16 materialization (the exact conversion site):
  `vendor/mlx-rocm-upstream/mlx/io/gguf.cpp:99-101` — `extract_tensor_data()`
  calls `gguf_tensor_to_f16(tensor)` and throws
  `"[load_gguf] gguf_tensor_to_f16 failed"` on NULL.
- Routing switch: `gguf.cpp:352-366` — Q2_K/Q5_K/Q6_K -> raw k-quant load,
  Q4_0/Q4_1/Q8_0 -> `gguf_load_quantized`, everything else ->
  `extract_tensor_data` (f16). IQ2_S (type 22) falls to the else.
- The pin's conversion table (`gguflib.c` at 8fa6eb65,
  `gguf_tensor_to_f16`): exactly F32/F16/BF16/Q8_0/Q4_K/Q6_K/Q2_K/Q4_0/Q4_1;
  any other
  type hits `else { free(f16); errno = EINVAL; return NULL; }`. Claim (1)
  VERIFIED: the card's fault string, type table, and ~5s pre-GEMM failure are
  all reproducible from source.
- Note (bonus finding): the pin's gguflib.c block-size table (line 42) says
  `q2_k 256, 82` — the WRONG Q2_K size (84 B is correct); upstream fixed it in
  c495778 (2026-05-16, "fix Q2_K block size"). Inert for the in-tree raw path
  (gguf_quants.cpp hardcodes 84), but the pin is pre-fix.

### (2) "upstream master has zero i-quants" — PARTIALLY INCORRECT; the
### actionable conclusion still holds
- Upstream master state, fetched + date-stamped 2026-11-26:
  `fdfafbed766db0a1e9019b07994cd88f133d1aab`, last commit 2026-05-16
  ("fix Q2_K block size").
- Master DOES carry i-quant type codes: `gguflib.h:36-42`
  (GGUF_TYPE_IQ2_XXS=16 .. GGUF_TYPE_IQ2_S=22) and the block-size table
  `gguflib.c:48-54` (`iq2_s 256, 82`). The pin (8fa6eb65) carries the same
  header constants (`gguflib.h:36-42`) — so a reader branch can use
  `GGUF_TYPE_IQ2_S` without a pin bump.
- What master lacks (and the pin lacks): any i-quant dequant-to-f16 function,
  and any i-quant branch in `gguf_tensor_to_f16` (master function body,
  verified: identical 9-type chain, no IQ branch).
- Conclusion: "a pin bump does not fix the loader" HOLDS, but the stated
  mechanism ("zero i-quants") does not; the correct statement is "no i-quant
  conversion function / no conversion-table branch upstream".

### (3) Control (mmproj BF16 loads) — CONSISTENT, not independently verifiable
- The F16/BF16 branches exist in the pin's conversion table; the failure
  is pinned to unsupported types only. Consistent with the code; no box
  access from this lane.

### (4) f16 materialization = ~54-58G for any 27B — VERIFIED structurally,
### with a scope correction
- Every non-quant-routed tensor lands as f16 via `gguf.cpp:99`
  (~2 B/weight -> 27e9 x 2 = 54 G). VERIFIED for the on-box wheel: the
  wheel on the 16G box (+75ca881, 2026-11-03) predates the WS3 k-quant
  stack, and the current vendor pin (aed6a25, 2026-11-13) `io/gguf.cpp`
  has NO k-quant branch — Q2_K on the pin goes through
  `gguf_tensor_to_f16` (Q2_K is in the pin's table) = ~54 G f16. The
  card's "~54G even for Q2_K" is accurate for the wheel of record.
- Scope correction: on origin/main (post-2026-11-20), Q2_K/Q5_K/Q6_K do NOT
  materialize f16 — `gguf_load_kquant_raw` (gguf_quants.cpp:177+) keeps the
  compressed stream (Q2_K 84 B/256 -> ~9.3 G). But those tensors have NO
  quantized-preserving GEMM yet either (see B-iii); the only consumers are
  `kquant_dequantize` (-> f32, worse) and `kquant_file_gemm` (no Q2_K).

### (5) in-tree rdna4 mmq IQ2_S kernels — VERIFIED verbatim
- `mmq-config-rdna4.cuh:198-209` = 12 `CASE(GGML_TYPE_IQ2_S, ...)` refs
  (also rdna3 = 11, gcn = 11, rdna3-5 = 12, rdna2 = 11);
  `dequantize_iq2_s` at `mmq/dequantize.cuh:310` (14-line device dequant);
  type traits `mmq/common.cuh:1123`; `block_iq2_s` struct
  `mmq/ggml-common.h:401` (82 B/256); full
  `mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>` instantiation
  `mmq/mmq-host.hip:132`; compiled into the wheel
  (`mlx/backend/rocm/CMakeLists.txt:184-185`).
- Bridge gap VERIFIED: the ONLY mmq-family python binding is
  `mmul_q_glu` (4-bit affine G64; `python/src/fast.cpp:674` and
  `python/src/fast.cpp` docstring "W is mlx 4-bit affine G64"). No
  primitive, op, or binding routes GGUF raw blocks (i-quant or k-quant) to
  the mmq kernels. The kernels sit behind ggml-backend plumbing
  (`ggml_backend_cuda_context` / `mmq_args`, `mmq/mmq.hip`), not behind any
  MLX primitive.

## B. Candidate paths to a 2.5-bit-class 27B resident in <= 14 G

Common precondition for ALL paths: re-pin the vendor/build input to a tree
carrying the WS3 + mmq stack (it landed in the 2026-11-20 vendor sync
f5e7cd8; the on-box wheel and the current pin predate it) + rebuild. That
re-pin+rebuild is one cost shared by (i) and (iii).

### (i) i-quant GGUF reader + route IQ2_S to the rdna4 mmq kernels — MEDIUM
What is missing, piece by piece:
1. Reader branch (SMALL): `gguf.cpp:352-366` add
   `GGUF_TYPE_IQ2_S` (and optionally IQ2_XS/XXS, IQ3_*) -> a new
   `gguf_load_iq2_raw` (raw uint8 `{nblocks, 82}`); the pattern already
   exists in `gguf_load_kquant_raw` (gguf_quants.cpp:177+). The type
   constants exist in the pinned gguflib.h:36-42 — NO gguflib pin bump or
   gguflib patch is needed (the raw path bypasses
   `gguf_tensor_to_f16` entirely).
2. GPU GEMM consuming raw IQ2_S blocks (the real work, two sub-options):
   - (b1) wire a new fast op to the EXISTING `mul_mat_q IQ2_S`
     instantiations: HEAVY — mmq is ggml-backend-shaped
     (`ggml_backend_cuda_context`, `mmq_args`, ggml tensor buffers); every
     call needs a ggml context + buffer bridging from MLX arrays.
   - (b2) extend the kquant_file-style in-kernel-dequant GEMM (the
     `kquant_file_gemm.hip` family) with an IQ2_S per-element dequant ported
     from `dequantize_iq2_s` (dequantize.cuh:310) + the `iq2s_grid` table
     (ggml-common.h) — BOUNDED; follows the existing per-ktype branch
     pattern (ktype 12/13/14 today; ktype 22 would be the 82-B/256 case).
3. Vehicle-side routing (SMALL-MEDIUM): reshape raw `{nblocks, 82}` ->
   `(K, N/256, 82)` and dispatch per-tensor, following
   `python/src/kquant_file.cpp`.
Delivers the ISTA quality class (IQ2_S 101.8% zero-shot recovery per the
terminal card's cited numbers). Effort class: MEDIUM (b2 option).

### (ii) MLX-format 2-bit affine build — NOT SUPPORTED in-tree; LARGE
- The tree's affine quantized matmul set is 4-bit / 8-bit (+ fp8); no 2-bit
  affine GEMM/GEMV for general linear layers exists (checked qmm/qmv kernel
  families: no 2-bit block configs).
- The ONLY 2-bit kernels in-tree are purpose-built:
  `backend/rocm/lm_head_int2.hip` and `lm_head_int2w.hip` — int2
  g128-asymmetric dequant GEMV (M=1 ONLY), fixed quarter-split packing,
  for the MTP draft head. Not general matmul; not a 27B path.
- No 2-bit quantize/convert op exists in-tree (no 4-bit->2-bit conversion
  utility found).
- Effort: LARGE (a new 2-bit affine GEMM family + conversion utility +
  vehicle), and 2-bit affine without an imatrix is a known-weak quality
  class — well below IQ2_S. NOT the cheapest; not recommended for the
  op's quality intent.

### (iii) Q2_K GGUF via the existing WS3 stack — SMALL-MEDIUM
- origin/main already has: reader branch (gguf.cpp:352-359),
  `gguf_load_kquant_raw` (Q2_K 84 B/256), and `kquant_file_gemm`
  (dequant-at-GEMM, quantized-preserving; HBM stays ~9.3 G for a 27B Q2_K).
- The gap: `kquant_file_gemm` does NOT support ktype 10 (Q2_K). Exact
  sites: `block_bytes_for` (kquant_file_gemm.hip:55) returns -1 for 10;
  `check_layout` error text (kquant_file_gemm.hip:838) omits it; per-ktype
  dequant branches (GEMM kernel ~:426-472; standalone dequant
  ~:671-698; gather ~:755-767) cover only 8/12/13/14.
- The Q2_K device dequant already exists in-tree: `kq_dequant_q2k`
  (kquants.hip:95) and `kquant_dequantize` supports ktype 10
  (fast.cpp:2452). Adding ktype 10 to the file-GEMM family = a bounded
  kernel addition + a G1 bit-exact check vs the GGUF oracle + vehicle
  wrapper.
- Quality caveat: Q2_K (2.625 bpw) is a low-quality 2-bit class (no
  imatrix; k-quant 2-bit is the known-weak corner of GGUF). It unblocks the
  HBM-fit + throughput machinery at the 2.5-bit-class size, but does not
  carry the ISTA IQ2_S quality bar.

## C. Reader-to-kernel gap for path (i) — exact sites that change
1. `vendor/mlx-rocm-upstream/mlx/io/gguf.cpp:352-366` (load_arrays): add
   the IQ2_S (22) branch -> new `gguf_load_iq2_raw` in
   `mlx/io/gguf_quants.cpp` (raw `{nblocks, 82}`, mirroring
   `gguf_load_kquant_raw:177+`; block geometry 256/82 is already in
   gguflib.h:42 of the pin).
2. New op + kernel: `mlx/fast.h` + `mlx/fast.cpp` (KQuantFileGemm-style
   class), a new `backend/rocm/quantized/*.hip` in-kernel-dequant GEMM with
   the IQ2_S element dequant ported from `mmq/dequantize.cuh:310`
   (+ `iq2s_grid` from `mmq/ggml-common.h`), python binding in
   `python/src/` (kquant_file.cpp pattern).
3. `python` vehicle wrapper: reshape + dispatch (kquant_file.cpp pattern).
4. NO gguflib change (raw path bypasses the conversion table); NO pin bump.
Alternative (b1) would instead add a ggml-context bridge op — the heavy
route; it is stated, not recommended.

## Verdict (code finding, not a strategy ruling)
- Cheapest decisive path for a 2.5-bit-class model resident in <= 14 G:
  (iii) Q2_K via the WS3 stack + the bounded ktype-10 kernel addition
  (plus the shared re-pin+rebuild).
- The path that carries the op's IQ2_S quality intent: (i), MEDIUM effort,
  with the bounded (b2) kernel option; the mmq IQ2_S kernels are already
  compiled in-tree, so the gap is a reader branch + a GEMM op + vehicle
  routing, not missing kernel science.
- (ii) is excluded on code evidence: no 2-bit affine matmul or 2-bit
  conversion exists in-tree; building it is LARGE and quality-weak.

## Check lines
- a. style-format (`uv run ruff format --check .`): PASS (106 files already
  formatted).
- b. persona-leak (standing pattern list over this card + branch diff):
  HIT COUNT: 2 (both on the required fleet-identity line; see below).
- c. line-wrap: `scripts/line_wrap_check.py` is ABSENT from the tree
  (checked workdir and origin/main) — not executable; card lines manually
  kept to <= 78 columns.

## Fleet identity
Fleetwork ticket 3549; git identity cloudyfromff7 (cloudy@ag.mrzk.io);
code reviewer 1/3, wave-12. Branch: wave12-code-review (mirror, never
origin/main).
