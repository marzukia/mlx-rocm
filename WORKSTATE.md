# WORKSTATE — GDN fused T=2 (MTP verify) — final state

Branch `worker/gdn-verify-fusion`. Fused GDN kernel extended to T=2 (S=2 MTP verify), model F-branch `S == 1` → `S <= 2`. Full details in `REPORT.md` (verdict first).

## State
- Gates (final wheel, guest-a): Gate 1 lossless A/B **PASS** (byte-identical tokens, 18816 fused calls, 68.21→72.69 tok/s); Gate 2 fused-AR **PASS** (80.9–81.1, equal to pre-T2 kernel in-same-env); Gate 3 MTP-on > 81.37 **NOT MET** (72.7 tok/s, 73.4% acceptance — see REPORT §4 for the ceiling analysis).
- Wheel: built from the vendor clone (`$MLX_VENDOR_DIR`, 75ca881 + PR3-HYBRID dirty + this graft), `python3 setup.py bdist_wheel`, installed in `venv-fused2`.
- Op probe 8/8 bit-exact (T=1/T=2 vs model default, invariance).

## Key facts for the next session
- Root cause of the first T=2 failure: `q3/qkv[:, :, :key_dim]` is a row-strided slice (row stride 8192 ≠ 2048); the kernel's packed read is wrong for row 1. Fixed by force-packed copies for T>1 only (T=1 is zero-overhead).
- TMAX template constant (1 or 2, at launch) is what keeps the AR path at the original codegen (a runtime t-loop cost ~0.3%).
- Fallback scope: T=1 → the 4-op chain (1-D-only, safe at T=1); T>1 with the fused path disabled → clean throw (the chain GPU-faults at T>1); T=0 → rejected. The T2 kill-switch is removed (A/B is FUSE=0 vs FUSE=1).
- `mx.eval()` in this build returns None (sync flush); use `np.asarray(x.astype(mx.float32))`.
- `MTP_SIDECAR` must be a basename (`model-mtp.safetensors`); the bench guard renames all sidecars except the match to `*.disabled`.

## Follow-ups (REPORT §7)
1. Gate 3: fuse the MTP draft module's 2-token forward or improve draft acceptance; re-measure.
2. Adversarial review of the patch set (reporter ≠ submitter), then merge.
