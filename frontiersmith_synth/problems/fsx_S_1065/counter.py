#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for
"Demolish the Bridge City, Pier by Pier".

<in>  : "n m" then m lines "u v" (1-indexed, symmetric brace pattern).
<out> : the participant's demolition order -- a permutation of 1..n (whitespace
        separated, any layout).
<ans> : unused placeholder.

Feasibility: the output must parse as exactly n integers, each an integer token in
[1, n], with no repeats (a permutation). Any violation (wrong count, duplicate,
out-of-range, unparseable / non-finite token) -> "Ratio: 0.0".

Objective: simulate symbolic Gaussian elimination of the sparse symmetric brace
pattern in the given pivot order. Eliminating a pier that still has d active braces
costs exactly d*(d+1)//2 scalar operations (d to release/rebalance the braces, plus
C(d,2) to install a temporary cross-brace between every still-standing pair that
was not already directly braced -- fill-in, which then persists as a real brace for
the rest of the process). Fewer total operations is better.
"""
import sys


def read_ints(path):
    with open(path, "r") as f:
        return f.read().split()


def parse_input(path):
    toks = read_ints(path)
    n = int(toks[0])
    m = int(toks[1])
    edges = []
    idx = 2
    for _ in range(m):
        u = int(toks[idx]) - 1
        v = int(toks[idx + 1]) - 1
        idx += 2
        edges.append((u, v))
    return n, edges


def ops_for_order(n, edges, order):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    alive = [True] * n
    ops = 0
    for v in order:
        nbrs = [u for u in adj[v] if alive[u]]
        d = len(nbrs)
        ops += d * (d + 1) // 2
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
        alive[v] = False
    return ops


def main():
    if len(sys.argv) < 3:
        print("usage: counter.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]

    n, edges = parse_input(in_path)

    try:
        out_toks = read_ints(out_path)
    except Exception:
        print("parse error reading output\nRatio: 0.0")
        return

    if len(out_toks) != n:
        print(f"wrong token count: got {len(out_toks)}, expected {n}\nRatio: 0.0")
        return

    order = []
    seen = [False] * n
    for tok in out_toks:
        # strict integer token: reject decimals / exponents / nan / inf / junk
        if not (tok.lstrip("+-").isdigit()):
            print(f"non-integer token: {tok!r}\nRatio: 0.0")
            return
        try:
            val = int(tok)
        except Exception:
            print(f"unparseable token: {tok!r}\nRatio: 0.0")
            return
        if val < 1 or val > n:
            print(f"pier id out of range: {val}\nRatio: 0.0")
            return
        zi = val - 1
        if seen[zi]:
            print(f"duplicate pier id: {val}\nRatio: 0.0")
            return
        seen[zi] = True
        order.append(zi)

    if len(order) != n or not all(seen):
        print("not a permutation of 1..n\nRatio: 0.0")
        return

    submitted_ops = ops_for_order(n, edges, order)
    baseline_ops = ops_for_order(n, edges, list(range(n)))

    ratio = min(1.0, 0.1 * baseline_ops / max(1e-9, float(submitted_ops)))
    print(f"n={n} m={len(edges)} baseline_ops={baseline_ops} submitted_ops={submitted_ops}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
