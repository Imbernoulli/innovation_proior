# TIER: strong
# INSIGHT: fitting a good codebook for THIS round's own data is not the hard part (an
# ordinary warm-started Lloyd pass, like the greedy tier's, already does that well).
# The hard part is spending the metered movement budget wisely. The greedy recipe
# scales EVERY slot's needed shift down by the same uniform factor -- so a slot that
# barely needs to move gets nudged a little (wasting budget: a codeword that ends up
# NEAR but not AT a useful position still gets out-competed by whichever OTHER slot is
# genuinely closest, so a partial nudge for every slot buys little), while a slot that
# has been left stranded far behind by the drift only gets partway there and stays
# nearly as useless as before. A slot moved ALL the way to its target immediately
# starts reconstructing nearby points well; the same total movement smeared thinly
# across every slot leaves every slot still losing the nearest-neighbour competition to
# wherever it would have gone anyway. So instead: rank slots by how far the drift has
# left them behind their current-round target, and spend the budget completing full
# relocations for as many of the most-stranded slots as it can afford end to end,
# leaving already-well-placed slots untouched.
import sys, json
import numpy as np
from scipy.optimize import linear_sum_assignment

inst = json.load(sys.stdin)
D, K = inst["D"], inst["K"]
pts = np.array(inst["points"], dtype=float)
n = pts.shape[0]
prev = inst.get("prev_codebook")
budget = inst.get("move_budget")


def kmeans_fit(points, K, init, iters=25):
    centers = init.copy()
    npts = points.shape[0]
    for _ in range(iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        newc = centers.copy()
        for k in range(K):
            mask = assign == k
            if mask.any():
                newc[k] = points[mask].mean(axis=0)
        if np.allclose(newc, centers, atol=1e-10):
            centers = newc
            break
        centers = newc
    return centers


# --- PCA + radius-quantile initialization (manifold-aware starting point) ---
mean = pts.mean(axis=0)
centered = pts - mean
if n > 1:
    cov = (centered.T @ centered) / n
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    top2 = evecs[:, order[:2]]
    radius = np.sqrt(np.sum((centered @ top2) ** 2, axis=1))
else:
    radius = np.zeros(n)
order_idx = np.argsort(radius, kind="stable")
init = np.zeros((K, D))
for k in range(K):
    lo, hi = (k * n) // K, ((k + 1) * n) // K
    grp = order_idx[lo:hi]
    init[k] = pts[grp].mean(axis=0) if len(grp) else pts[order_idx[min(lo, n - 1)]]

target = kmeans_fit(pts, K, init)

if prev is None:
    codebook = target
else:
    prevarr = np.array(prev, dtype=float)
    cost = np.linalg.norm(prevarr[:, None, :] - target[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    target_aligned = np.zeros_like(target)
    target_aligned[row_ind] = target[col_ind]

    shifts = target_aligned - prevarr
    need = np.linalg.norm(shifts, axis=1)
    order_need = np.argsort(-need)  # largest relocation need first

    codebook = prevarr.copy()
    remaining = float(budget)
    for k in order_need:
        d = need[k]
        if d <= 1e-12:
            continue
        if remaining >= d:
            codebook[k] = target_aligned[k]
            remaining -= d
        elif remaining > 1e-12:
            frac_k = remaining / d
            codebook[k] = prevarr[k] + frac_k * shifts[k]
            remaining = 0.0
        else:
            break

diffs = pts[:, None, :] - codebook[None, :, :]
d2 = np.sum(diffs ** 2, axis=2)
assign_out = np.argmin(d2, axis=1).astype(int).tolist()

print(json.dumps({"codebook": codebook.tolist(), "assign": assign_out}))
