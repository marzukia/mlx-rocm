W3.1 vehicle per the pin ruling (cloudy, 2026-11-21): per-chunk release + the restored 8GB/1GB defaults, on the W4 tree (e0885604, PR #46 head). Stacked slice on PR #46.

What changes (all in vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/mmq/mmq.hip; vendor.json re-pin):
(1) Cap defaults restored to the original P1B23 values: the wq4 repack default 2GB -> 8GB (8589934592) and the xq81 default 256MB -> 1GB (1073741824) at the g_wq4_cache / g_xq81_cache constructor sites. The env hooks (MLX_ROCM_MMQ_REPACK_MAX_BYTES / MLX_ROCM_MMQ_XQ81_MAX_BYTES) and their SI parse stay UNCHANGED (the A/B reference; the W4 caps stay reachable via the overrides, =2147483648 / =268435456). Label note on record: 'W3 class' is loose (W3 carried no caps - the P1B4 per-call variant, symbols absent); the VALUES 8GB/1GB are the pin.
(2) Per-chunk release of the device-resident mmq scratch after each prefill chunk (the harness prefill-chunk class, M >= 256; the single 4K forward = no wrap = no release; decode M < 256 excluded): the persistent P1B20/P1B23 caches + the LRU evict-oldest structure + the kqfg shim stay; the release empties both caches once per chunk. Trigger = the weight-identity sequence wrap (the first per-chunk weight re-touched), the identity-keyed repack-cache path only. DESIGN CONSTRAINT (the t3116 sync-class gate): the release adds NO per-dispatch syncs - it runs ONCE per chunk: one hipDeviceSynchronize (drain) + a hipFree loop (device idle, the internal free-syncs are cheap). Per-chunk is ~1/8 the t3116 1992-syncs-per-4K-pass count class, bounded + A/B measurable (MLX_ROCM_MMQ_RELEASE_PER_CHUNK=0 = the persistent-only A/B reference).
(3) Proof hook: the [pcr] emit under MLX_MMQ_CFG (one line per chunk: nbuf + syncs=1) + the T3317-PCREL-20261121 canary marker for the strings check (alongside REPACK_MAX_BYTES / XQ81_MAX_BYTES / Q4K_ALLOC). No new kernel family: allocation-path config only.

Ticket t3317 + the pin ruling (cloudy, 2026-11-21) cited.

Build (helium, the t3309 pattern: a0ctl cmake + make, the build-trap class carried: clean relink + delete stale objects + rebuild + md5 + strings):
W3.1 wheel md5 c649197a3ffd69028f2a9cc0217c0f2f (mlx-0.32.3.dev20261121+t3317dev1, drop t3317dev1.whl), libmlx.so md5 680bd3d54ab6e3cb3c37468212477ece, core .so md5 10c1b83112d5c85c9565def3a61c2dcc (unchanged from W4 - no binding change). DISTINCT vs all 5 prior builds: W4 wheel e62aa901ff877f14efb0bb78a1e6fab2 / lib 06a5a5735ce8c3dfa7061003794dc7f4, W3 3e5f23f1 / 129e12d6, 7be 5359fe46 / 7be573f0, 265055a 982f9981 / 51653411. strings on the built lib: REPACK_MAX_BYTES / XQ81_MAX_BYTES / Q4K_ALLOC / RELEASE_PER_CHUNK / T3317-PCREL-20261121 all present.

Gates (in progress on helium; card to follow):
- the canary before any arm (single-4K peak + 8xCH512, no host shadow, rc=0)
- the t3116 sync-class proof (one round, W4 installed vehicle = the control vs W3.1): the [cfgq4k] dispatch count parity + the [pcr] once-per-chunk count + the e2e class (no 3.2x drag)
- the 27B MTP K=3 fit + rate re-confirm (prewarm + 4 measured rounds, the t3298 gate harness, int2 head caveat on record D1/D2/D3 96.9/85.5/98.1, TPK 3.609)

worker: monky, fresh-agent check to follow
persona-leak check: 3 hits (the worker line + the two pin-ruling cites)
