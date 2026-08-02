#!/usr/bin/env python3
# Deterministic checker for "Dual-Channel Relay Mesh" (format C, maximize clique suppression).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys
from itertools import combinations


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def build_adj(n, color, c):
    adj = [0] * n
    for i in range(n):
        row = color[i]
        for j in range(i + 1, n):
            if row[j] == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def count_mono_cliques(adj, n, k):
    cnt = 0
    for combo in combinations(range(n), k):
        ok = True
        for a in range(len(combo)):
            row = adj[combo[a]]
            for b in range(a + 1, len(combo)):
                if not (row >> combo[b]) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            cnt += 1
    return cnt


def total_violations(n, k, color):
    adj0 = build_adj(n, color, 0)
    adj1 = build_adj(n, color, 1)
    return count_mono_cliques(adj0, n, k) + count_mono_cliques(adj1, n, k)


def block_baseline_violations(n, k):
    # Checker's own trivial construction: partition vertices into consecutive groups of
    # size (k-1) (last group possibly smaller) so no group alone can contain a k-clique.
    # Edges inside a group -> channel 0. Edges between groups gi,gj -> channel (gi+gj)%2.
    # This is a naive, non-clique-aware deterministic rule (NOT random, NOT algebraic).
    g = max(1, k - 1)
    color = [[0] * n for _ in range(n)]
    for i in range(n):
        bi = i // g
        for j in range(i + 1, n):
            bj = j // g
            c = 0 if bi == bj else (bi + bj) % 2
            color[i][j] = c
            color[j][i] = c
    return total_violations(n, k, color)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        n = int(itoks[0]); k = int(itoks[1])
    except Exception:
        fail("bad instance")

    if n < 2 or k < 3 or k > n:
        fail("bad instance params")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = n * n
    if len(otoks) != need:
        fail("expected %d tokens (n x n channel matrix), got %d" % (need, len(otoks)))

    color = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            tok = otoks[idx]; idx += 1
            if tok not in ("0", "1"):
                fail("entry (%d,%d)='%s' is not 0/1" % (i, j, tok))
            color[i][j] = int(tok)

    for i in range(n):
        for j in range(i + 1, n):
            if color[i][j] != color[j][i]:
                fail("asymmetric entries (%d,%d)" % (i, j))

    # Objective: V = exact count of monochromatic k-cliques (both channels). Fewer is better.
    V = total_violations(n, k, color)

    B = block_baseline_violations(n, k)
    B = max(1, B)
    V = max(1, V)

    sc = min(1000.0, 100.0 * B / V)
    print("n=%d k=%d V=%d B=%d Ratio: %.6f" % (n, k, V, B, sc / 1000.0))


if __name__ == "__main__":
    main()
