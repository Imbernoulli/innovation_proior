# TIER: strong
# Iterative low-rank matrix completion (SoftImpute / EM-PCA) with SELF-TUNED rank.
# The true cohort matrix is low-rank plus noise, so completing the missing cells with a
# rank-r reconstruction recovers the shared physiological factors and beats both the
# column mean and nearest-neighbour rules. The catch is choosing r: too small under-fits
# the signal, too large over-fits the measurement noise. So we pick r by INTERNAL
# CROSS-VALIDATION -- hide a deterministic 1/7 of the OBSERVED cells as a validation
# fold, run the completion at each candidate rank, and keep the rank with the smallest
# validation error -- then refit on all observed cells. The measurement-noise floor
# keeps the error strictly positive (headroom).
import sys, json, math
import numpy as np

inst = json.load(sys.stdin)
N, D = inst["N"], inst["D"]
M = inst["matrix"]
mask = inst["masked"]

cmean = np.zeros(D)
csd = np.ones(D)
for j in range(D):
    vals = [M[i][j] for i in range(N) if M[i][j] is not None]
    if vals:
        a = np.asarray(vals, dtype=float)
        cmean[j] = a.mean()
        s = a.std()
        csd[j] = s if s > 1e-12 else 1.0

obs = np.zeros((N, D), dtype=bool)
Z = np.zeros((N, D), dtype=float)
for i in range(N):
    for j in range(D):
        if M[i][j] is not None:
            obs[i, j] = True
            Z[i, j] = (M[i][j] - cmean[j]) / csd[j]


def softimpute(observed_mask, rank, iters=80):
    F = Z.copy()
    F[~observed_mask] = 0.0
    for _ in range(iters):
        F[observed_mask] = Z[observed_mask]
        try:
            U, sv, Vt = np.linalg.svd(F, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        r = min(rank, len(sv))
        recon = (U[:, :r] * sv[:r]) @ Vt[:r, :]
        F[~observed_mask] = recon[~observed_mask]
    return F


# deterministic validation fold from the observed cells
obs_idx = [(i, j) for i in range(N) for j in range(D) if obs[i, j]]
st = 12345
keys = []
for _ in range(len(obs_idx)):
    st = (st * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    keys.append((st >> 17) % 1000000)
perm = sorted(range(len(obs_idx)), key=lambda k: keys[k])
n_val = max(D, len(obs_idx) // 7)
val_cells = [obs_idx[perm[t]] for t in range(min(n_val, len(obs_idx)))]
obs_tr = obs.copy()
for (i, j) in val_cells:
    obs_tr[i, j] = False

best_rank, best_err = 1, float("inf")
for rk in range(1, min(D - 1, 8) + 1):
    F = softimpute(obs_tr, rk)
    acc = 0.0
    for (i, j) in val_cells:
        d = F[i, j] - Z[i, j]
        acc += d * d
    err = acc / len(val_cells)
    if err < best_err:
        best_err, best_rank = err, rk

F = softimpute(obs, best_rank)
preds = [float(F[i, j] * csd[j] + cmean[j]) for (i, j) in mask]
print(json.dumps({"preds": preds}))
