#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for quotient-balanced-partition.

Feasibility: exactly n integer tokens, each in [1,k], each plot's tree count
exactly matches the prescribed size.

Objective (minimize): mean, over every row (window) test and every graft
(log-index-mod-d) test, of the WORST-PLOT absolute deviation between actual
and proportional count. Score = min(1000, 100*B/max(1e-9,F)) / 1000 where B
is the same statistic computed on the checker's own reference partition
(consecutive-position blocks).
"""
import sys


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def find_primitive_root(p):
    n = p - 1
    fac = factorize(n)
    for g in range(2, p):
        ok = True
        for q in fac:
            if pow(g, n // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None  # unreachable for prime p


def build_idx(p, g):
    """idx_arr[x] = discrete log of residue x (1<=x<=p-1) base g."""
    n = p - 1
    idx_arr = [0] * (p)  # idx_arr[0] unused
    val = 1
    for i in range(n):
        idx_arr[val] = i
        val = (val * g) % p
    return idx_arr


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    p = int(nxt())
    k = int(nxt())
    sizes = [int(nxt()) for _ in range(k)]
    m_row = int(nxt())
    windows = [(int(nxt()), int(nxt())) for _ in range(m_row)]
    m_graft = int(nxt())
    grafts = [int(nxt()) for _ in range(m_graft)]
    return p, k, sizes, windows, grafts


def worst_plot_dev_window(labels, sizes, n, k, t, w):
    counts = [0] * (k + 1)
    for x in range(t, t + w):
        counts[labels[x]] += 1
    worst = 0.0
    for i in range(1, k + 1):
        expected = sizes[i - 1] * w / n
        worst = max(worst, abs(counts[i] - expected))
    return worst


def worst_plot_dev_graft(labels, sizes, n, k, idx_arr, d):
    counts = [[0] * d for _ in range(k + 1)]
    for x in range(1, n + 1):
        counts[labels[x]][idx_arr[x] % d] += 1
    worst = 0.0
    for i in range(1, k + 1):
        expected = sizes[i - 1] / d
        for r in range(d):
            worst = max(worst, abs(counts[i][r] - expected))
    return worst


def objective(labels, sizes, n, k, windows, grafts, idx_arr):
    devs = []
    for (t, w) in windows:
        devs.append(worst_plot_dev_window(labels, sizes, n, k, t, w))
    for d in grafts:
        devs.append(worst_plot_dev_graft(labels, sizes, n, k, idx_arr, d))
    return sum(devs) / len(devs)


def reference_labels(sizes, k, n):
    labels = [0] * (n + 1)
    x = 1
    for i in range(1, k + 1):
        for _ in range(sizes[i - 1]):
            labels[x] = i
            x += 1
    return labels


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    p, k, sizes, windows, grafts = read_instance(in_path)
    n = p - 1

    with open(out_path) as f:
        toks = f.read().split()

    if len(toks) != n:
        print(f"Feasibility: wrong token count {len(toks)} != {n}  Ratio: 0.0")
        return 0

    labels = [0] * (n + 1)
    for x in range(1, n + 1):
        tok = toks[x - 1]
        try:
            c = int(tok)
        except ValueError:
            print(f"Feasibility: token {x} not an integer  Ratio: 0.0")
            return 0
        if c < 1 or c > k:
            print(f"Feasibility: label {c} at position {x} out of range [1,{k}]  Ratio: 0.0")
            return 0
        labels[x] = c

    counts = [0] * (k + 1)
    for x in range(1, n + 1):
        counts[labels[x]] += 1
    for i in range(1, k + 1):
        if counts[i] != sizes[i - 1]:
            print(f"Feasibility: plot {i} has {counts[i]} trees, expected {sizes[i-1]}  Ratio: 0.0")
            return 0

    g = find_primitive_root(p)
    idx_arr = build_idx(p, g)

    F = objective(labels, sizes, n, k, windows, grafts, idx_arr)

    ref = reference_labels(sizes, k, n)
    B = objective(ref, sizes, n, k, windows, grafts, idx_arr)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"F={F:.6f} B={B:.6f} Ratio: {ratio:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
