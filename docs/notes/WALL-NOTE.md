# WALL NOTE (lane state, 2026-11-20)

3.7K PP prefill lane — #36 ONE-PR consolidated stack merged to main `f5e7cd88` (2026-11-19 22:37Z) + #37 pad, by op call.

1. Parity merge-gate CLEARED: fixed-protocol record 3719.25 t/s @4K = 100.98% of the 3683.1 bar (the external 500x sustained stream). The fixed record protocol (22:2xZ) never changes. Axis records: 3084.9 t/s @4K (the f2c zero-mode axis, x3, the FLAG-OFF record state; record vehicle lib 070122d8) / 2861.4 t/s (the pcm bench class).
2. Flags: MLX_SDPA_FATTN ON-by-default in the merged tree (op letter (a) = ADOPT) - the flag-ON vs flag-OFF boost A/B rides the parity re-stamp. MLX_ROCM_INPROJ_PAD default OFF (e2e unmeasured; no win claimed).
3. Pending: the [314] gfx1201-box restoration (sysfs clock window + the L1 f2c re-control on 070122d8, 2 rounds, expecting 3084.9 +/-2%).
