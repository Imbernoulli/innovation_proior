# TIER: greedy
# The obvious first recipe, run FRESH each round (this program has no memory across
# rounds -- only `prev_codebook`, which it treats purely as a movement anchor, never as
# information about the manifold): fit an ordinary ambient-space K-means to THIS
# round's own points only, then drag the previous codebook's slots toward the
# freshly-fit local centroids, scaling the whole move down uniformly if it would
# exceed move_budget. This never asks whether the curve seen in earlier rounds still
# matters, so once the active window has drifted far from wherever the codebook
# currently sits (and the movement budget is tight), it permanently lags -- and even
# when it does catch up locally, it wastes the budget re-discovering coverage the
# stream already walked through.
import sys, json, math

inst = json.load(sys.stdin)
D, K = inst["D"], inst["K"]
pts = inst["points"]
prev = inst.get("prev_codebook")
budget = inst.get("move_budget")
n = len(pts)


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans_local(points, K, init_centers, iters=18):
    centers = [list(c) for c in init_centers]
    n = len(points)
    assign = [0] * n
    for _ in range(iters):
        for i, p in enumerate(points):
            bd, bk = None, 0
            for k, c in enumerate(centers):
                d = dist2(p, c)
                if bd is None or d < bd:
                    bd, bk = d, k
            assign[i] = bk
        newc = [[0.0] * D for _ in range(K)]
        cnt = [0] * K
        for i, p in enumerate(points):
            k = assign[i]
            cnt[k] += 1
            for d in range(D):
                newc[k][d] += p[d]
        for k in range(K):
            if cnt[k] > 0:
                for d in range(D):
                    newc[k][d] /= cnt[k]
            else:
                newc[k] = centers[k]
        centers = newc
    return centers


if prev is None:
    if n >= K:
        srt = sorted(range(n), key=lambda j: pts[j][0])
        step = n / K
        init = [pts[srt[int(k * step)]] for k in range(K)]
    else:
        init = [pts[k % n] for k in range(K)]
    centers = kmeans_local(pts, K, init)
else:
    target = kmeans_local(pts, K, prev)
    shifts = [[target[k][d] - prev[k][d] for d in range(D)] for k in range(K)]
    total = sum(math.sqrt(sum(v * v for v in shifts[k])) for k in range(K))
    scale = 1.0 if total <= budget or total <= 1e-12 else budget / total
    centers = [[prev[k][d] + scale * shifts[k][d] for d in range(D)] for k in range(K)]

assign_out = [0] * n
for i, p in enumerate(pts):
    bd, bk = None, 0
    for k, c in enumerate(centers):
        d = dist2(p, c)
        if bd is None or d < bd:
            bd, bk = d, k
    assign_out[i] = bk

print(json.dumps({"codebook": centers, "assign": assign_out}))
