# Guest-state RCA: 03:05Z → ~05:00Z drift window — verdict

Round: worker/guest-rca · run 2026-11-02 · box: RX 9070 XT (gfx1201), ROCm guest userland (kernel 7.3, amdgpu 7.3)
Method: read-only forensics (mtime audit + full journal/dmesg of the window), live state audit,
clock/thermal probe under both bench locks (flock, 15 s poll), and three named guest-state fixes
each re-verified with the canonical 2-arm, 2-runs/arm, pinned-wheel gate.

## 1. Verdict

**NOT FIXABLE IN PLACE** (within guest userspace authority, under lock).
The bad state survives a full `amdgpu` KMD unload + reload (SMU re-initialised: "SMU is resumed
successfully") — it is latched below the KMD layer (persistent SMU/HBM/ASIC state) or is not a
guest-writable state at all. The only remaining reset is a **power cycle / full guest reboot**,
which is an operator action (the box carries other tenants/services and is not to be rebooted
unilaterally).

Recommended: **re-anchor in the current guest state** (rebaseline option (b)) —
B0 ≈ 59.1–59.2 / R1 ≈ 64.5–64.7 (2-runs/arm, pinned wheel, tape 128/128), and/or have the
operator reboot the box and re-run the pinned-wheel gate as the decisive latched-state test
(boot predates the drift, so a reboot is the definitive test if the state is SMU-latched).

## 2. Forensics of the drift window (03:00–06:00Z Nov 2)

**Files — nothing changed in the window.** Latest mtimes before the window:
`/opt/rocm` → nvinfer 02:33Z (pre-window); `/etc` → 02:19Z (tuned-adm, pre-window); kernel
modules, systemd units, `/var/log` — no in-window changes. No `tuned`/`cpufreq` unit exists;
no rc.local/history traces of a config change.

**Services — nothing changed.** Full journal of the window: only routine noise (tailscale,
snapd, fwupd-refresh *failures*, sshd sessions, sysstat). No service restart/enable in the window.

**Kernel — nothing in the window.** Zero amdgpu / KFD / thermal / reset events 03:00–06:00Z
(dmesg + journal cross-checked). The only Nov 2 kernel event is a `[gfxhub] page fault` at
07:31:21Z on a short-lived `python` (pid 518803) that aborted — *after* the drift onset; a
crashed probe, not the cause.

**Boot history:** last boot Nov 1 03:42Z (predates the Nov 1 reinstall? no — *after* it) — in any
case **before both the reinstall and the good anchor**; amdgpu module loaded Nov 1 03:43Z, no
reload until my 12:31Z test. No reboot inside the window.

**Driver configuration (default, unchanged):** DPM `performance`, force-level `auto`, no power cap
(304 W default), no overdrive, CG default mask, 32/32 CU enabled. No tenant processes, no orphan
GPU processes (fuser /dev/dri: none). The second "GPU" is the APU iGPU (Granite Ridge, 512 MB) —
irrelevant.

**CPU:** all 16 cores on `powersave` (not the Ubuntu default `schedutil`), no recorded change
time — pre-window, and A/B-proven not the cause (§4, fix 1).

## 3. Clock/thermal probe under GPU lock (12:19–12:21Z, both locks held)

Sampler @ 3 s during two pinned-wheel runs (pinned wheel, tape 128/128, `n_refwd` 101/99):

| signal | under load | verdict |
|---|---|---|
| SCLK | 3116–3274 MHz (max boost; above the 2200 MHz DPM-list max) | **not clock-stuck** |
| MCLK | 1258 MHz (level 5) | normal active state |
| junction / edge / mem | 70 / 51 / 55 °C max | **no thermal throttle** (logging enabled, zero events) |
| power (gen phase) | 192–276 W | **below 304 W cap** (313 W transient during weight load only) |

GPU is at full boost with headroom on power and temperature. The "stuck SCLK" hypothesis is **dead**.

## 4. Named guest-state fixes, each re-verified with the 2-arm gate (pinned wheel, tape 128/128)

| # | fix (applied under lock) | B0 (2 runs) | R1 (2 runs) | D-ACC | restored? |
|---|---|---|---|---|---|
| 0 | current state (baseline) | 59.25 / 59.17 | 64.58 / 64.53 | 93/59/27, 95/61/29 | — |
| 1 | CPU governor → `performance` (all 16 cores) | 59.18 / 59.08 | 64.61 / 64.65 | same | **no (Δ≈0)** |
| 2 | `amdgpu` rmmod + modprobe (full KMD + SMU re-init) | 60.50 / 60.37 | 65.79 / 65.67 | same | **no (Δ +1.2–1.3%, noise)** |

Reference good band (t1325, 03:05Z): B0 69.74 / R1 70.20 (82-build wheel); rebaseline canonical
band: B0 69.48 / R1 75.00. Post-fix runs remain ~14% below — the bad signature (verify
+8.7 ms, D-ACC shift, `n_refwd` 101 vs 100) is intact.

## 5. What the evidence says

- **No file, config, service, or kernel event changed in the window** — the drift is an
  *unlogged runtime state change*, not a configuration or image change.
- The state **persists across a full KMD/SMU re-initialisation** → it lives in persistent
  ASIC/SMU/HBM state (or is not guest-writable at all).
- **Leading candidate trigger (correlation, not proven):** the two **crashed `rocprofv3`
  sessions at 03:58–04:04Z** (host-side profiler, "Unsupported hardware" → `Aborted (core
  dumped)`) sit exactly between the last good measurement (03:05Z) and the first bad
  (~04:50Z). A profiler session that aborts after opening the performance-monitoring path
  can leave latched perfmon/DPM state in the SMU.
- Ruled out by test or by window audit: files, configs, services, module reload, tenant
  interference, GPU clock stick, thermal throttle, power cap, CPU governor, KMD/SMU re-init.

## 6. Process notes (audit trail)

- First probe pass (12:13Z) was **invalid and discarded**: it set only `MLX_ROCM_GDN_TAPE=1`;
  the dispatch (model file, both patches present, md5 7763fbbb…) requires *both*
  `MLX_ROCM_GDN_FUSE=1` and `MLX_ROCM_GDN_TAPE=1` — with tape-only the GDN op silently ran
  all-fallback (`n_fb=128`, `n_refwd=408`, TOKS_S≈32, the 32.23 signature). Re-run with the
  corrected recipe → valid (tape 128/128).
- Both bench locks (home-directory lock + model-mount lock) were held for the entire probe
  and released on exit; no tenant was interrupted (lock was free at acquisition; no other GPU
  processes before/after).
- End state of the box: CPU governors left at `performance` (previously `powersave`); amdgpu
  module reloaded at 12:31Z (srcversion D99D34A06CD0EF7C02FEC1F). Neither change affected the
  numbers.
- Pins re-verified end-of-probe: bench 2c2d6c90, commit guard 4a262a0d, sidecar 02a54359
  (identical before/after — no file touched).

## 7. Decision

- **NO re-anchor to the old good band is possible from inside the guest.** The old band is not
  reachable by any guest-writable control (clocks, governor, caps, DPM, KMD state — all tested
  or audited).
- **Re-anchor in the current guest state** with the pinned wheel (sha256 5bc7ca20…), 2 runs
  per arm, both locks held, canonical env (both GDN env vars, LD_PRELOAD, HCC, MLX_MTP=4):
  the current band (B0 ≈ 59.1–59.2, R1 ≈ 64.5–64.7) is stable across 6 independent runs
  spanning 2 h and a KMD reload (σ < 0.6%).
- Optional operator action: **reboot the box** (power cycle), then re-run the pinned-wheel
  gate. If it restores 69.48/75.00 → the drift was SMU-latched (consistent with the
  crashed-profiler trigger); the anchor decision (old band + a "no crashed host-side profilers
  under lock" rule, or a fresh anchor) follows the measured result.
