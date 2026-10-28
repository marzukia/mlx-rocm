# t789 — vendor-pin fix (dangling pin + silent build-wheel) — worker report

Branch: `worker/t789-vendor-pin-fix` (base `origin/main` @ 64b98b8).
Fixes the t781 review-FAIL finding on PR #10 (96a1677 "fix(rocm): GDN fused
decode T=1 launcher pointer write"): `vendor.json` pinned
`75ca881b2e82f7b4069c8564a203792aa16d8a29`, but the fork branch
`lifeofpiboy/mlx:tail-fusion-gdn` was at `96a1677` (the fix's parent) —
`75ca881` existed only in a guest build clone and was never pushed, and
`scripts/build-wheel.sh` never verified the checkout against the pin.

## 1. Push the pin (part 1)

- `75ca881` recovered from the t761 build clone (`/mnt/models/mlx-clone-g4`;
  the task text said g1, but g4 is the clone that carries it — g1's HEAD is
  `1641333`). Fetched into a local repo, then:

  ```
  git push --force-with-lease origin 75ca881:refs/heads/tail-fusion-gdn
  #   To github.com:lifeofpiboy/mlx.git
  #    + 96a1677...75ca881 75ca881 -> tail-fusion-gdn (forced)
  ```

  Fast-forward on the branch (96a1677 is 75ca881's parent; the push is a
  2-commit advance, no history rewrite of the pinned commit).

- Verification (all three agree):
  - `git ls-remote https://github.com/lifeofpiboy/mlx.git refs/heads/tail-fusion-gdn`
    → `75ca881b2e82f7b4069c8564a203792aa16d8a29`
  - `gh api repos/lifeofpiboy/mlx/commits/75ca881b2e82f7b4069c8564a203792aa16d8a29`
    → 200, `sha=75ca881b...`, `commit.message="fix(rocm): GDN fused decode T=1 launcher pointer write"`,
    parent `96a16773a1133795406a85545b654b6972d93721`
  - `gh api repos/lifeofpiboy/mlx/branches/tail-fusion-gdn` →
    `sha=75ca881b...`
  - `75ca881` is also in the fork's default branch (`main`) history
    (`git merge-base --is-ancestor` → yes), so the pin is resolvable from
    any clone of the fork.

## 2. build-wheel.sh: verify the checkout against the pin (part 2)

`scripts/build-wheel.sh` now reads the pin from `vendor.json`
(`PIN_SHA=$(python3 -c 'import json; print(json.load(open("vendor.json"))["ref"])')`)
and, after the clone/checkout, verifies it:

```bash
# Verify the checkout matches the pinned sha — a dangling pin must fail
# loudly, not build a different commit silently.
CHECKED_OUT_SHA=$(git rev-parse HEAD)
if ! git cat-file -e "${PIN_SHA}^{commit}" 2>/dev/null; then
  echo "ERROR: pinned sha ${PIN_SHA} not resolvable in the vendor clone" >&2
  echo "       (pin=$PIN_SHA checked-out=$CHECKED_OUT_SHA)" >&2
  exit 1
fi
if [ "$(git rev-parse "${PIN_SHA}^{commit}")" != "$CHECKED_OUT_SHA" ]; then
  echo "ERROR: checkout does not match vendor.json pin" >&2
  echo "       (pin=$PIN_SHA checked-out=$CHECKED_OUT_SHA)" >&2
  exit 1
fi
echo "vendor checkout verified: ${CHECKED_OUT_SHA} == vendor.json pin ${PIN_SHA}"
```

Behavior (exercised with a stubbed clone, three branches):
- pin not resolvable in the clone → `ERROR ... not resolvable` + exit 1
- pin resolvable but checkout differs → `ERROR ... does not match vendor.json
  pin (pin=... checked-out=...)` + exit 1
- pin matches → `vendor checkout verified: ... == vendor.json pin ...` + continue

No other change to the script (clone/fetch/install flow untouched).

## 3. G1/G2 on a wheel built from the landed ref (part 3)

Built on neon (guest, ROCm 10.1, gfx1201) under `flock /mnt/models/gpu-bench.lock`:

- Fresh clone of `lifeofpiboy/mlx` @ `tail-fusion-gdn` →
  `/mnt/models/mlx-clone-t789`; `git rev-parse HEAD` =
  `75ca881b2e82f7b4069c8564a203792aa16d8a29` (the pinned commit, from the
  fork — not the guest build clone).
- `MLX_VENDOR_DIR=/mnt/models/mlx-clone-t789 bash scripts/build-wheel.sh`
  (with the new verification — it printed `vendor checkout verified` and
  proceeded) → `/mnt/models/dist-t789/mlx-0.32.3.dev20261028+75ca881-cp312-...whl`.
- Installed into a scratch venv (`/mnt/models/venv-t789`, python 3.12.3;
  `venv-g4` untouched). The scratch venv mirrors venv-g4's mlx_lm model
  files: `qwen3_5.py` (S==1-guarded fused branch) **and** `gated_delta.py`
  (K3 patch: non-fused GDN update routes through
  `mx.fast.gated_delta_update` on ROCm). The first gate round ran with the
  stock PyPI `gated_delta.py` (Python op loop) and G2 diverged at token 33 —
  an environment artifact, not a wheel difference: the m0 reference and the
  t771 AR baseline were both produced with the K3-patched wrapper. With the
  K3 file in place, all gates below pass.

### Rule-6 effective config (printed + asserted by `scripts/t789_config.py`)

```
t789-config: OK
  env: MLX_GDN_TAIL_FUSION=1  MLX_ROCM_GDN_FALLBACK=0  MLX_GDN_MLXFAST=1
  wheel: 0.32.3.dev20261028+75ca881  (source sha 75ca881b2e82f7b4069c8564a203792aa16d8a29)
  device: gpu  (rocminfo: gfx1201, AMD Radeon RX 9070 XT)
```

Kernel names (rocprofv3 `--kernel-trace` over a dispatch of all three fused
ops): `gdn_prep`, `gdn_qk_norm`, `gdn_norm_gated` all dispatched.

### G1 — no-fault serve/generation: PASS

`TAIL_FUSION=1`, greedy, 1024 tokens, qwen3-9b-mlx 4bit:
`FUSION=1 LOAD=2.82 GEN=14.71 TOKS=1024 TOKS_S=69.59`, rc=0, no HSA-fault
signature in the log (grep `HSA|fault|Aborted|core dumped|Segmentation|HIP`
→ no match).

### G2 — maxdiff 0.0 vs the 4bit-MLX reference + fused==unfused: PASS

Same model/prompt/seed as the t612 reference (bench4 `m0.out`, 1024 greedy
tokens), vacuity-guarded stream compare:

```
G2-FUSED-VS-M0-REF:    STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
G2-FUSED-VS-UNFUSED:  STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
GATE-LIVE (all compared streams non-empty)
```

The fused stream is bit-identical to both the 4bit-MLX (m0) reference and
the same-wheel unfused stream. Cross-check: the t771 AR baseline
(`bench-g4d/m0a.out`, 75ca881 wheel, venv-g4) is byte-identical to m0
(diff 0/1024), so the reference itself is stable across the two 75ca881
builds.

### t761 primitive gates (re-run on this wheel): PASS

```
PRIM-GATES-PASS
FWD-LAYER maxdiff=0.000000e+00            (fused T=1 layer fwd, 5-layer chain)
[zero]/[rand] norm_gated maxdiff=0.0      (gdn_norm_gated vs (x/||x||)*z)
[zero]/[rand] qn,kn,vc,beta,g maxdiff=0.0 (gdn_prep vs conv1d+silu+rms_norm+sigmoid+compute_g; g float32)
```

### Sibling state (GPU lock)

Lock was free for every t789 run. Sibling t784 (QMV lane, dist-g4e):
`build_g4e3.log` shows the `ced8c58` wheel build completed 12:23 UTC
(sha256 `d85065b3...` recorded) — no GPU process of theirs was active at
check time; no wall-clock remainder visible.

## 4. Files

- `scripts/build-wheel.sh` — pin verification (this PR)
- `scripts/t789-run.sh`, `scripts/t789_config.py`, `scripts/t789_gen.py`,
  `scripts/t789_kernels.py` — gate harness (kept in-repo for re-run; all
  skip-safe off-GPU)
- `REPORT.md` — this report

# t771: Speed bench on the fixed gdn wheel (dist-g4d @75ca881)

MTP-ON 100 tok/s lane. Measured the fixed wheel (t761: gdn_prep launcher fix,
14 kernel-arg host `Data*` ptrs passed as device ptrs; fix commit `75ca881`).
G1/G2 had already PASSed on dist-g4d (no-fault, maxdiff 0.0 vs reference,
fused==unfused). This run is the first speed measurement on the fixed wheel.

## Verdict (gates, operational order 2026-10-28)

| Gate | Result |
|---|---|
| 1. Lossless (G4, m2 vs m0, #8 harness) | **FAIL** — first divergent step **149** (1-based), 865/1024 tokens differ |
| 2. Beat AR (64.94 pre-fix baseline) | MTP 64.34 vs AR 69.65 → **FAIL** (-7.6% vs AR) |
| 3. Enable MTP default | **OFF** (gate 1 failed; gate 2 failed) |

Lane goal 100 tok/s decode: **not reached**. AR moved to 69.65 tok/s; MTP K=1
M=2 is 64.34 tok/s @ 76.0% accept — the gdn fix did **not** close the MTP gap.

## Phase 1 — pre-stage + verify (all PASS)

- SSH chain: `ssh -J andryo@100.91.94.56 neonvm@192.168.122.234` (guest neonvm).
- Build clone: `git -C /mnt/models/mlx-clone-g4 log --oneline -1`
  → `75ca881 t761: fix gdn_prep launcher - data_shared_ptr().get() is a host Data*, not a device pointer`
- Wheel identity (venv `/mnt/models/venv-g4`, python3.12):
  `mlx-0.32.3.dev20261028+75ca881` (+ `mlx_lm-0.32.2`) — matches the fixed
  clone ref. `import mlx.core` OK under `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`.
- Lock: `flock -w 10 /mnt/models/gpu-bench.lock -c true` → free (no GPU siblings).
- Model-load smoke (flock, 10 tokens, Qwen3.5 GDN-hybrid 9B 4bit
  `/mnt/models/qwen3-9b-mlx`): `SMOKE-OK n=10 dt=1.53`, no fault,
  gfx1201 bound.

## Phase 2 — bench (flock, fresh process per run, 3 repeats, median)

Harness: repo `bench/mtp-bench3.py` (AR = `MLX_MTP=0`, MTP K=1 M=2 =
`MLX_MTP=2`), 1024 tokens greedy, smartphone prompt, run via new runner
`bench/bench-g4d.sh` (copied to guest `/mnt/models/tmp/bench-g4d/`):

```
flock -w 300 /mnt/models/gpu-bench.lock env MLX_MTP=<0|2> \
  /mnt/models/venv-g4/bin/python /mnt/models/tmp/bench-g4d/mtp-bench3.py
```

| Config | n | min | **median** | max | accept |
|---|---|---|---|---|---|
| AR (`MLX_MTP=0`) | 3 | 69.31 | **69.65** | 69.75 | — |
| MTP K=1 M=2 (`MLX_MTP=2`) | 3 | 63.92 | **64.34** | 64.46 | 389/512 = 76.0% |

Raw lines (identical accept across all 3 m2 repeats):

```
MTP=0 LOAD=2.76 GEN=14.68 TOKS=1024 TOKS_S=69.75 ACCEPTED=0/0
MTP=0 LOAD=2.78 GEN=14.70 TOKS=1024 TOKS_S=69.65 ACCEPTED=0/0
MTP=0 LOAD=2.78 GEN=14.78 TOKS=1024 TOKS_S=69.31 ACCEPTED=0/0
MTP=2 LOAD=2.77 GEN=16.02 TOKS=1024 TOKS_S=63.92 ACCEPTED=389/512
MTP=2 LOAD=2.79 GEN=15.89 TOKS=1024 TOKS_S=64.46 ACCEPTED=389/512
MTP=2 LOAD=2.79 GEN=15.92 TOKS=1024 TOKS_S=64.34 ACCEPTED=389/512
```

**AR delta vs pre-fix (dist-g4c @96a1677): 69.65 vs 64.94 → +4.71 tok/s (+7.2%).**
The gdn fix changed the base path (gdn_prep runs on every target forward,
including AR). Pre-fix numbers are delta-only, not current.

MTP vs AR on the fixed wheel: 64.34 vs 69.65 → **-5.31 tok/s (-7.6%)**.
Pre-fix the gap was -1.8% (63.8 vs 64.94); the fix widened it. Accept rate
unchanged (76.0% vs 76% pre-fix).

MTP K=2 check: **not run** — conditional on (b) >= AR; 64.34 < 69.65.

## Phase 3 — G4 lossless gate (#8 harness, vacuity-guarded)

`GATE-LIVE` (all compared streams non-empty, 1024 tokens each).

| Compare | Status |
|---|---|
| G4-M2-M0 (m2a vs m0a, exact) | **FAIL** first_div=149 match=159 diff=865 of 1024 |
| M2-A-B, M2-B-C (m2 determinism) | MATCH 1024/1024 |
| M0-A-B, M0-B-C (m0 determinism) | MATCH 1024/1024 |

First divergent step: **149** (1-based). Tokens 145-160:

```
m0 (AR): 4779,11,321,5090,24296,13,271,794,357,13,561,717,383,29900,318,4272
m2 (MTP): 4779,11,321,5090,33443,51010,82,13,271,794,357,13,561,717,383,29900
```

m2 inserts 2 extra tokens (`51010, 82`) at 150-151 before re-syncing at 271 —
a state-advance/rollback mismatch in the MTP verify path (reject path:
GDN save/restore + KV trim + re-forward). Both paths are run-to-run
deterministic, so this is systematic, not noise.

**G4 verdict: FAIL (lossless-only; drift at step 149).**

Note for follow-up (not diagnosed here): the harness `save_gdn` zero-copy
save (`conv, ssm, lengths` per GDN layer) predates the fix; if the fixed
gdn_prep uses an additional state buffer, the MTP rollback would silently
miss it. The pre-fix t612 run reported m2 byte-identical to m0, so the
divergence is specific to the fixed wheel's state layout/path.

## Phase 4 — harness change

`bench/bench-g4d.sh` (new): parameterized runner (venv/harness/out overridable,
defaults to venv-g4), 3 repeats per config with median, conditional K=2, and
the PR #8 vacuity-guarded lossless gate (same `cmp_streams` helper as
`bench4.sh`/`bench5.sh`). No changes to `mtp-bench3.py`/`mtp-bench5.py`.

## Exact commands (repro)

```
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234
git -C /mnt/models/mlx-clone-g4 log --oneline -1        # 75ca881
ls /mnt/models/venv-g4/lib/python3.12/site-packages/ | grep mlx
flock -w 10 /mnt/models/gpu-bench.lock -c true          # LOCK-FREE
# smoke (10 tokens, flock -w 240): venv-g4 python, load + 10x argmax step
scp bench/{mtp-bench3.py,mtp-bench5.py,bench-g4d.sh} neonvm@...:/mnt/models/tmp/bench-g4d/
bash /mnt/models/tmp/bench-g4d/bench-g4d.sh            # 6 runs + gate (K2 not run)
```
