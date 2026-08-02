# TIER: greedy
# Textbook "SVDimpute" / hard-impute low-rank matrix completion (Troyanskaya et al.-style):
# fill missing entries with a row-mean warm start, repeatedly replace them with a
# fixed-rank SVD reconstruction of the current filled matrix, keep observed entries fixed.
# This is the standard nuclear-norm-flavoured recipe; it never looks at the side-info
# graphs and treats every missing cell as if it were sampled uniformly at random.
import sys, math, random


def power_iterate(M, n, m, iters, rng):
    v = [rng.uniform(-1.0, 1.0) for _ in range(m)]
    nv = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / nv for x in v]
    u = [0.0] * n
    for _ in range(iters):
        u = [sum(M[i][j] * v[j] for j in range(m)) for i in range(n)]
        nu = math.sqrt(sum(x * x for x in u)) or 1e-12
        u = [x / nu for x in u]
        v = [sum(M[i][j] * u[i] for i in range(n)) for j in range(m)]
        nv = math.sqrt(sum(x * x for x in v)) or 1e-12
        v = [x / nv for x in v]
    Mv = [sum(M[i][j] * v[j] for j in range(m)) for i in range(n)]
    sigma = sum(u[i] * Mv[i] for i in range(n))
    return u, sigma, v


def rank_r_reconstruct(M, n, m, r, iters, rng):
    resid = [row[:] for row in M]
    comps = []
    for _ in range(r):
        u, sigma, v = power_iterate(resid, n, m, iters, rng)
        comps.append((u, sigma, v))
        for i in range(n):
            for j in range(m):
                resid[i][j] -= sigma * u[i] * v[j]
    recon = [[0.0] * m for _ in range(n)]
    for (u, sigma, v) in comps:
        for i in range(n):
            for j in range(m):
                recon[i][j] += sigma * u[i] * v[j]
    return recon


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); L = int(next(it)); seed = int(next(it))
    p = int(next(it))
    mask = [[False] * m for _ in range(n)]
    val = [[0.0] * m for _ in range(n)]
    row_sum = [0.0] * n
    row_cnt = [0] * n
    tot_sum, tot_cnt = 0.0, 0
    for _ in range(p):
        i = int(next(it)); j = int(next(it)); v = float(next(it))
        mask[i][j] = True
        val[i][j] = v
        row_sum[i] += v
        row_cnt[i] += 1
        tot_sum += v
        tot_cnt += 1
    re_ = int(next(it))
    for _ in range(re_):
        next(it); next(it)
    ce_ = int(next(it))
    for _ in range(ce_):
        next(it); next(it)
    q = int(next(it))
    queries = []
    for _ in range(q):
        i = int(next(it)); j = int(next(it))
        queries.append((i, j))

    global_mean = tot_sum / tot_cnt if tot_cnt > 0 else 0.0
    row_mean = [(row_sum[i] / row_cnt[i]) if row_cnt[i] > 0 else global_mean for i in range(n)]

    filled = [[val[i][j] if mask[i][j] else row_mean[i] for j in range(m)] for i in range(n)]

    r = min(6, n, m)
    rng = random.Random(12345)
    for _ in range(12):
        recon = rank_r_reconstruct(filled, n, m, r, 25, rng)
        filled = [[val[i][j] if mask[i][j] else recon[i][j] for j in range(m)] for i in range(n)]

    out = ["%.4f" % filled[i][j] for (i, j) in queries]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
