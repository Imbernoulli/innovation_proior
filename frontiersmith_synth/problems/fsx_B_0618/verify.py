#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the gemstone-sticker
unfolding problem. Prints "Ratio: <float in [0,1]>" on its last line and exits 0.
"""
import sys
import math

DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]
OPP = {(0, 2), (2, 0), (1, 3), (3, 1)}


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 3:
        fail("input truncated")
    n = int(toks[0]); m = int(toks[1]); penalty = int(toks[2])
    idx = 3
    edges = []
    for _ in range(m):
        if idx + 4 > len(toks):
            fail("input edge list truncated")
        u = int(toks[idx]); su = int(toks[idx + 1])
        v = int(toks[idx + 2]); sv = int(toks[idx + 3])
        idx += 4
        edges.append((u, su, v, sv))
    return n, m, penalty, edges


def read_output(path, m):
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) == 0:
        fail("empty output")
    # strict finiteness / integer parsing
    def parse_int(tok):
        try:
            fv = float(tok)
        except Exception:
            fail("non-numeric token '%s'" % tok)
        if not math.isfinite(fv):
            fail("non-finite token '%s'" % tok)
        if fv != int(fv):
            fail("non-integer token '%s'" % tok)
        return int(fv)

    k = parse_int(toks[0])
    if k < 0 or k > m:
        fail("k out of range")
    if len(toks) != 1 + k:
        fail("token count mismatch: expected %d got %d" % (1 + k, len(toks)))
    kept = []
    seen = set()
    for t in toks[1:1 + k]:
        e = parse_int(t)
        if e < 0 or e >= m:
            fail("edge index out of range")
        if e in seen:
            fail("duplicate edge index")
        seen.add(e)
        kept.append(e)
    return kept


def layout(n, edges, kept):
    """Place every face; returns (pos, comps) or None if a cycle is detected."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx in kept:
        u, su, v, sv = edges[idx]
        ru, rv = find(u), find(v)
        if ru == rv:
            return None  # cycle -> not a forest
        parent[ru] = rv

    adj = [[] for _ in range(n)]
    for idx in kept:
        u, su, v, sv = edges[idx]
        adj[u].append((v, su, sv))
        adj[v].append((u, sv, su))

    pos = [None] * n
    rot = [None] * n
    comp = [-1] * n
    comps = []
    for start in range(n):
        if comp[start] != -1:
            continue
        cid = len(comps)
        comp[start] = cid
        pos[start] = (0, 0)
        rot[start] = 0
        stack = [start]
        cells = [start]
        while stack:
            u = stack.pop()
            for (v, su, sv) in adj[u]:
                if comp[v] != -1:
                    continue
                g = (su + rot[u]) % 4
                pos[v] = (pos[u][0] + DX[g], pos[u][1] + DY[g])
                rot[v] = (g + 2 - sv) % 4
                comp[v] = cid
                stack.append(v)
                cells.append(v)
        comps.append(cells)
    return pos, comps


def bbox_cost(pos, comps, penalty):
    total = 0
    for cells in comps:
        xs = [pos[u][0] for u in cells]
        ys = [pos[u][1] for u in cells]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        total += area
    total += penalty * (len(comps) - 1)
    return total


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, penalty, edges = read_input(in_path)
    kept = read_output(out_path, m)

    result = layout(n, edges, kept)
    if result is None:
        fail("kept edges contain a cycle (not a forest)")
    pos, comps = result

    # self-overlap-avoidance: no two facets of the SAME piece may land on the
    # same unit grid cell.
    for cells in comps:
        seen = set()
        for u in cells:
            p = pos[u]
            if p in seen:
                fail("self-overlapping net in one piece")
            seen.add(p)

    F = bbox_cost(pos, comps, penalty)

    # internal baseline B: the trivial construction that cuts every fusion
    # edge, shipping each of the n facets as its own 1x1 sticker.
    B = n * 1 + penalty * (n - 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("pieces=%d F=%.6f B=%.6f" % (len(comps), F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
