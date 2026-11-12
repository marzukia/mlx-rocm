# W1 PHASE 2 REPRO - the 108 number, run from the in-tree files (2026-11-12, jimmy)

Op order: the PR must carry the repro procedure for the 108 number, and the repro must be RUN from the in-tree files (not the box copies). The in-tree run (part 2 of this order, on the card after the G2 session ends) is the proof that the in-tree files - including the scrubbed bench - reproduce the number, i.e. the one named scrub delta is functionally inert on the 108 path.

## The number (G1 record battery: seq S=3 @1024, the B config, 3 rounds)

- rate: 107.90 / 108.42 / 108.68 t/s across the 3 G1 rounds (the 108 t/s class)
- TOK_PER_STEP 2.248, 1151 tokens per round
- acceptance D1ACC 390 / D2ACC 249 (D1PCT 76.2 / D2PCT 63.8), D3ACC 0
- REFWDS 0, GDN_SEQ_CALLS 12288 (24 layers x 512 verifies)
- stream 64b1b6da in all 3 rounds (md5 of the ALLTOKS token list, space-joined, no prefix, no trailing newline)
- FIRST20: 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279

## Provenance (the G1-verified states, no drift)

- wheel: sha256 392103cafaedf387b5d94544aba85b51d75756b3de86c6f1a8df1feac2622853 (mlx-0.32.3.dev20261112+3eb5b6f, cp314); the installed lib md5 is ceef7ef1 (core a579fdae) - the G1-PASS tree build (cloudy-ruled: 64b1b6da lineage, S=2 three-way byte-identity, T-independence bit-exact)
- model: in-tree models/qwen3_5.py = the G1 venv model, md5 303b245b, verbatim (the MTP6_MOD_BF16 guarded default-off cast + the slot2/seq model branches)
- bench: in-tree bench/k2arm-pinned.py = the G1 bench box state 4bf76a7d with ONE named scrub delta: the MTP_Q4K_GGUF env fallback default changed from an absolute operator path to the relative perclass-probe/Qwen3.5-9B-Q4_K_M.gguf (that arm is default-off behind MTP_Q4K=1 + CH_MTP_4BIT=1 and is not on the 108 path) - in-tree md5 9041ac2a
- tree head: 361dfaf (w1-phase1-bf16-20261112, PR #30)
- the model root must carry the MTP sidecar (md5 02a54359, the 15 f32 keys) - it resolves from MODELS, same as in the G1 run

## Run (the repro command)

On the card box, as the card user, one GPU job at a time (the driver flocks the shared card lock, 300s wait):

    bash bench/gdn_seq1024_repro.sh

What the driver does, in order:

1. installs the in-tree model (models/qwen3_5.py) over the venv site-packages copy and prints its md5 (expect 303b245b),
2. sets the record-config env verbatim from the G1 battery: MLX_MTP=3 MTP6_TAPE=seq MTP6_DRAFT_HEAD=bf16 MTP6_HEAD_H=postnorm MTP6_MOD_BF16=1, plus MTP_MAX_TOKENS=1024 per round,
3. runs bench/k2arm-pinned.py (the in-tree bench) for 3 rounds, DPM AUTO (unpinned; the sclk state is recorded per round, never pinned),
4. prints one REPRO-R line per round: rc, the stream md5, the MTP= line (rate/TPK/acceptance), REFWDS, FIRST20, the clock state.

## Env mapping (neutral in-tree form -> the box run state)

| in-tree (driver) | box run state |
|---|---|
| MODELS (default /models) | the models mount containing qwen3-9b-mlx (the G1 run set MODELS to that mount) |
| W1P2_VENV (default $HOME/venv-p2seq) | the venv holding the wheel above |
| W1P2_GPU_LOCK (default $HOME/gpu.lock) | the shared card lock (flock; one GPU job at a time) |
| HCC_AMDGPU_TARGET=gfx1201, HIP_VISIBLE_DEVICES=0, LD_PRELOAD=libhsa-runtime64.so.1, TMPDIR=/var/tmp/scratch | the box run env (all neutral values, unchanged) |

## EXPECTED OUTPUT (the gate; same session, DPM AUTO, clock state stated verbatim per round)

- stream 64b1b6da in all 3 rounds
- 1151 tokens per round
- TOK_PER_STEP 2.248, D1ACC 390 / D2ACC 249, REFWDS 0
- TOKS_S within the G1 same-session band 107.9-108.7 (the G1 rounds were 107.90 / 108.42 / 108.68)
