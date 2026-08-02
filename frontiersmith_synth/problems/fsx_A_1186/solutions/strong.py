# TIER: strong
# Insight: the row/column similarity graphs reveal the (unlabeled) row- and column-group
# membership exactly (connected components). Once groups are known, the model is a
# two-way additive ANOVA: value = block[rowgroup][colgroup] + row_bias + col_bias.
# Fit it by block coordinate descent over the OBSERVED entries only, then evaluate the
# fitted block/bias terms on the missing cells -- including inside a fully-unobserved
# block, because that (rowgroup,colgroup) pair is still represented by the rows/columns
# of the same two groups that were NOT swallowed by the missing rectangle. This is what
# a pattern-blind low-rank/nuclear-norm fit (the "greedy" tier) cannot see: it has no
# notion of discrete group identity, only the raw observed-entry coordinates.
import sys
from collections import defaultdict


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); L = int(next(it)); seed = int(next(it))
    p = int(next(it))
    observed = []
    tot_sum, tot_cnt = 0.0, 0
    for _ in range(p):
        i = int(next(it)); j = int(next(it)); v = float(next(it))
        observed.append((i, j, v))
        tot_sum += v
        tot_cnt += 1
    re_ = int(next(it))
    row_parent = list(range(n))
    for _ in range(re_):
        a = int(next(it)); b = int(next(it))
        union(row_parent, a, b)
    ce_ = int(next(it))
    col_parent = list(range(m))
    for _ in range(ce_):
        a = int(next(it)); b = int(next(it))
        union(col_parent, a, b)
    q = int(next(it))
    queries = []
    for _ in range(q):
        i = int(next(it)); j = int(next(it))
        queries.append((i, j))

    row_comp = [find(row_parent, i) for i in range(n)]
    col_comp = [find(col_parent, j) for j in range(m)]

    global_mean = tot_sum / tot_cnt if tot_cnt > 0 else 0.0
    block = defaultdict(lambda: global_mean)
    row_bias = [0.0] * n
    col_bias = [0.0] * m

    for _ in range(40):
        sum_b = defaultdict(float); cnt_b = defaultdict(int)
        for (i, j, v) in observed:
            key = (row_comp[i], col_comp[j])
            sum_b[key] += v - row_bias[i] - col_bias[j]
            cnt_b[key] += 1
        for key, s in sum_b.items():
            block[key] = s / cnt_b[key]

        sum_r = [0.0] * n; cnt_r = [0] * n
        for (i, j, v) in observed:
            sum_r[i] += v - block[(row_comp[i], col_comp[j])] - col_bias[j]
            cnt_r[i] += 1
        row_bias = [(sum_r[i] / cnt_r[i]) if cnt_r[i] > 0 else row_bias[i] for i in range(n)]

        sum_c = [0.0] * m; cnt_c = [0] * m
        for (i, j, v) in observed:
            sum_c[j] += v - block[(row_comp[i], col_comp[j])] - row_bias[i]
            cnt_c[j] += 1
        col_bias = [(sum_c[j] / cnt_c[j]) if cnt_c[j] > 0 else col_bias[j] for j in range(m)]

    out = []
    for (i, j) in queries:
        pred = block[(row_comp[i], col_comp[j])] + row_bias[i] + col_bias[j]
        out.append("%.4f" % pred)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
