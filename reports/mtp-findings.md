# an earlier run partial findings (salvage 2026-10-27, session ended mid-turn at wall 2574s)

RCA of F1: MTP draft uncorrelated with target (rank ~206k-248k/248320, corr ~-0.02..-0.05).
Status: numpy reference BUILT but NOT validated. E3/E4 ground-truth experiments NEVER RAN.

## Established during an earlier run
1. **Norm convention**: the mlx port uses RMSNorm weights = HF_raw + 1 (i.e. the 4bit model
   files store (w-1)). Verified on 4 trunk norms (input 0.0331->1.0331, post -0.1126->0.8875,
   L5 0.1194->1.1195, final 1.1398->2.1398). The original numpy port used HF raw -> 8.5x
   residual attenuation (h1 norm 33 vs mlx 280), t1=220 instead of 271.
2. **MTP norm weights are wrong in the file the probe loads.** Table (mean values):
   | key                                | HF_raw | HF+1   | loaded(4bit) | mtpfile(on disk) |
   | mtp.layers.0.input_layernorm.w     | 0.0991 | 1.0991 | 0.0991       | 0.0495           |
   | mtp.layers.0.post_attn_layernorm.w | 0.2413 | 1.2413 | 0.2413       | 0.1207           |
   | mtp.layers.0.self_attn.q_norm.w    | 0.7573 | 1.7573 | 0.7573       | 0.3786           |
   | mtp.layers.0.self_attn.k_norm.w    | 0.7445 | 1.7445 | 0.7445       | 0.3723           |
   mtpfile = EXACTLY HALF of HF_raw. Per trunk convention the mathematically correct value is
   HF+1. So the MTP module's norms (input_layernorm, q_norm, k_norm) are suspect: check the
   in-memory values during a REAL mlx run (f1step.py path: load_model glob-loads
   model-mtp.safetensors + probe manually overwrites mtp.* from the same file) and compare to
   what the math requires. STRONG CANDIDATE for F1.
3. **After +1 norm fix**, numpy trunk gave t1=10992/t2=169261 (mlx: 271/8160) - numpy port
   STILL not validated; per-layer rnorms now diverge (numpy L0=183 vs mlx=29; L31 numpy=294
   vs mlx=714). Numpy port has a remaining bug (or the mlx trunk deviates from HF truth).
4. **L0 GDN comparison** (last completed experiment; mlx "K3 kernel" = mx.fast.gated_delta_update
   on ROCm vs numpy op-ref):
   - emb match (3.096 vs 3.093, maxdiff 0.0112), input_layernorm match (66.160 vs 66.158, 0.0039)
   - z close (611.054 vs 613.200, maxdiff 0.865), h1 close (maxdiff 1.1694)
   - **r (GDN out_proj output): K3 kernel norm=26.170 vs numpy 183.335, maxdiff=53.335** - LARGE
   - **ipz: mlx_shape=(4096,512) vs np_shape=(4096,4096)** - shape mismatch; likely probe
     extraction bug, not model. Verify before trusting #4's divergence.
   - K3 kernel vs op loop vs numpy: the three-way comparison was the in-flight work.

## Open (continuation must resolve)
- A: validate numpy port (t1/t2 = 271/8160, h1 cos > 0.99 vs mlx) - find remaining bug
- B: MTP in-memory norm values vs required (item 2) - fix model-mtp.safetensors if wrong
- C: E3 (numpy MTP fed (h1_ref,e1_ref) -> draft rank of t2_ref) and E4 (fed (h1_mlx,e1_mlx))
  - the ground truth that splits "mlx trunk corrupt" vs "convention wrong" vs "mlx MTP path corrupt"
- D: L0 GDN r-divergence (item 4) - real kernel bug or probe bug (ipz shape!)

## Assets
- np-rca-np.py: numpy trunk+MTP (modes: track/e1/mtp/e3/e4), bf16 shards, f32 math, layer-by-layer < 12GB
- np-rca-mlx-l0.py / -norms.py / -perlayer.py / -mtpnorms.py / -normconv.py: mlx-side probes
- Guest: ssh -o BatchMode=yes -J operator@<tailnet-ip> the build VM@<guest-ip>
  (scripts also live in /tmp on the guest; HF shards /mnt/models/hf-qwen3-9b/, mlx venv /mnt/models/venv-a,
  4bit model /mnt/models/qwen3-9b-mlx, mtp file model-mtp.safetensors 973MB f32)
- E2 mlx reference: t1=271('\n\n') t2=8160('Here'); target top8: [(8160,18.88),(248068,18.38),(760,16.25),...]
- numpy prompt top8 (unvalidated port): [(220,10.44),...] (pre +1-fix)
