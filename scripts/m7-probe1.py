import pickle
import numpy as np
import mlx.core as mx

def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None

# ---- 1. verify normalize_qk formula ----
import mlx_lm.models.gated_delta as gd
rng = np.random.default_rng(0)
x = (rng.standard_normal((1, 2, 16, 128)) * 3).astype(np.float32)
xm = mx.array(x)
inv_scale = 128.0 ** -0.5
eps = 1e-6
qn, kn = gd.normalize_qk(xm, xm, inv_scale=inv_scale, eps=eps)
rms_eps = eps * inv_scale ** 2
def rmsnorm(xv, e):
    return xv / np.sqrt(np.mean(xv * xv, axis=-1, keepdims=True) + e)
q_manual = (inv_scale ** 2) * rmsnorm(x, rms_eps)
k_manual = inv_scale * rmsnorm(x, rms_eps)
dq = npf(qn) - q_manual
dk = npf(kn) - k_manual
print("normalize_qk vs manual: q n=%d/%d maxabs=%.3e | k n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(dq != 0), dq.size, np.abs(dq).max(),
    np.count_nonzero(dk != 0), dk.size, np.abs(dk).max()))

# ---- 2. pkl: layer 4 row 0 ----
with open("/mnt/models/tmp/t880/cap-m7-s62.pkl", "rb") as f:
    c = pickle.load(f)
ca = c["gdnA"][4]
cb = c["gdnB"]["4_1"]
conv_rec = ca["conv1d"][0]
nq_rec = ca["normalize_qk"][0]
gp_rec = cb["gdn_prep"][0]

raw = npf(conv_rec["out"])[0, 0]  # (8192,) pre-silu
s = raw / (1.0 + np.exp(-raw))    # silu
q = s[:2048].reshape(16, 128)
k = s[2048:4096].reshape(16, 128)
q_manual = (inv_scale ** 2) * rmsnorm(q, rms_eps)
k_manual = inv_scale * rmsnorm(k, rms_eps)

aq = npf(nq_rec["out"][0])[0, 0]
ak = npf(nq_rec["out"][1])[0, 0]
bq = npf(gp_rec["out"][0]).reshape(16, 128)
bk = npf(gp_rec["out"][1]).reshape(16, 128)
print("A norm_q row0 vs manual: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(aq - q_manual != 0), aq.size, np.abs(aq - q_manual).max()))
print("A norm_k row0 vs manual: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(ak - k_manual != 0), ak.size, np.abs(ak - k_manual).max()))
print("B norm_q row0 vs manual: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(bq - q_manual != 0), bq.size, np.abs(bq - q_manual).max()))
print("B norm_k row0 vs manual: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(bk - k_manual != 0), bk.size, np.abs(bk - k_manual).max()))
print("A vs B norm_q row0: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(aq - bq != 0), aq.size, np.abs(aq - bq).max()))
print("A vs B norm_k row0: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(ak - bk != 0), ak.size, np.abs(ak - bk).max()))

# ---- 3. is A conv1d out itself wrong? manual conv from A conv1d in ----
xin = npf(conv_rec["in"][0])  # (1, 5, 8192)
w = npf(gp_rec["in"][8])      # (8192, 4) conv1d weight
print("conv in shape:", xin.shape, "weight shape:", w.shape)
manual_conv = np.zeros((2, 8192), dtype=np.float32)
for t in range(2):
    for j in range(4):
        manual_conv[t] += w[:, j] * xin[0, t + 3 - j]
d = manual_conv[0] - raw
print("A conv1d out row0 vs manual numpy conv: n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(d != 0), d.size, np.abs(d).max()))

# ---- 4. what does B gdn_prep receive as conv_state / qkv? ----
# B gdn_prep in: q (1,1,2048), k (1,1,2048), v (1,1,4096), b, a, A_log, dt_bias,
#               conv_state (3, 8192), conv1d.weight (8192, 4)
bq_in = npf(gp_rec["in"][0])  # (1, 1, 2048) — q = first 2048 of qkv
print("B gdn_prep q in: shape", bq_in.shape)
# B's qkv = the qkv input to gdn_prep? gdn_prep takes q,k,v separately (pre-conv)
# manual B conv: out = sum_j w[:, j] * [conv_state, qkv][t + 3 - j]
b_state = npf(gp_rec["in"][7])  # (3, 8192)
b_qkv = np.concatenate([npf(gp_rec["in"][0]).reshape(-1),
                        npf(gp_rec["in"][1]).reshape(-1),
                        npf(gp_rec["in"][2]).reshape(-1)])  # (8192,)
manual_b = np.zeros(8192, dtype=np.float32)
for j in range(4):
    manual_b += w[:, j] * (b_qkv if 3 - j == 3 else b_state[3 - j])
s_b = manual_b / (1.0 + np.exp(-manual_b))
qb = s_b[:2048].reshape(16, 128)
kb = s_b[2048:4096].reshape(16, 128)
qb_manual = (inv_scale ** 2) * rmsnorm(qb, rms_eps)
kb_manual = inv_scale * rmsnorm(kb, rms_eps)
print("B norm_q row0 vs manual(conv from B inputs): n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(bq - qb_manual != 0), bq.size, np.abs(bq - qb_manual).max()))
print("B norm_k row0 vs manual(conv from B inputs): n=%d/%d maxabs=%.3e" % (
    np.count_nonzero(bk - kb_manual != 0), bk.size, np.abs(bk - kb_manual).max()))
