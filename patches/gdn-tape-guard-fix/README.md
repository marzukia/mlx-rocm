# gdn-tape-guard-fix

Bench-side fix for the GDN state-tape commit guard (t1280 / gpu-verify2). The guard in
`gdn_tape_commit.py` read the live SSM state as `[Hv, 128, 128]` but the live leaf is
batch-led `[1, Hv, 128, 128]`; three checks were affected:

1. expected state-tape shape: `(t, c[1].shape[0]) + S_HEAD` → `(t,) + c[1].shape[1:]`;
2. live SSM tail check: `c[1].shape[1:]` → `c[1].shape[2:]` (+ ndim check);
3. rebind: `c[1] = st[row]` → `c[1] = st[row:row+1]` (keep the batch axis), same for conv.

Files: `gdn_tape_commit.py.stock` (as shipped with the bench), `gdn_tape_commit.py.fixed`,
`fix.diff`.

Measured effect on marzuki-neon (K=3, 128 verify, FUSE=1, TAPE=1, see REPORT-gpu-verify2.md):
n_ok 0/128 → 128/128, n_refwd 395→98 (vs 272 for the B0 control), refwd 36.5→9.0 ms/pair,
tok/s 38.1→61.1 (+34 % vs control).

Known remaining defect (op-side, not fixed here): the conv state-tape rows 0/1 are not
bit-exact with the S=1 chain (rows 2/3 and all SSM rows are), so the tape arm's stream
diverges from B0 and it is not servable until that is fixed. Also the tape op pybind
rejects `state=None` (fresh-cache first step).
