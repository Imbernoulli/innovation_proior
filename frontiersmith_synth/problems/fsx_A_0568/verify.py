#!/usr/bin/env python3
# Deterministic checker for "Calder Atelier: Torque-Balanced Mobile" (format C).
#
# The participant outputs a full binary "mobile": a rooted tree whose leaves carry
# a fixed multiset of integer weights and whose internal nodes are rods with a left
# arm aL and a right arm aR.  A rod is in static balance iff  wL*aL == wR*aR  (torque
# about the pivot), where wL,wR are the total weights hanging on each side.
#
# Balance fixes ONLY the ratio aL/aR = wR/wL, so the *magnitude* of every rod is a
# free integer scale (aL,aR)=(t*wR/g, t*wL/g), g=gcd(wL,wR).  The objective rewards a
# WIDE, EVENLY-spread silhouette (bin entropy of leaf horizontal positions) plus a
# balanced depth profile (entropy of leaf depths).  Balancing with minimal arms
# leaves the scale freedom on the table and clusters the silhouette near the spine.
#
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored). Prints "... Ratio: <r>".
import sys, math
from math import gcd
from collections import Counter

FLOOR = 0.11  # baseline floor -> keeps the score ceiling open (no saturation)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    N = int(toks[0]); Amax = int(toks[1])
    ws = [int(t) for t in toks[2:2 + N]]
    if len(ws) != N:
        raise ValueError("bad instance")
    return N, Amax, ws


# ---------------------------------------------------------------- objective
def objective(leaf_pos, N, Amax):
    """leaf_pos: list of (x, depth). F = 0.6*horizontal-bin-entropy + 0.4*depth-entropy."""
    xs = [p[0] for p in leaf_pos]
    ds = [p[1] for p in leaf_pos]
    binw = max(1, Amax // (2 * N))
    lN = math.log(N)
    cb = Counter(x // binw for x in xs)
    Hx = -sum((c / N) * math.log(c / N) for c in cb.values()) / lN
    cd = Counter(ds)
    Hd = -sum((c / N) * math.log(c / N) for c in cd.values()) / lN
    return 0.6 * Hx + 0.4 * Hd


# ---------------------------------------------------------------- reference baseline
# A complete balanced tree over a fixed leaf order with MINIMAL arms, then a
# deterministic collision-repair that bumps rod scales until every leaf is distinct.
# (Identical construction to solutions/trivial.py, so trivial reproduces this baseline.)
class RNode:
    __slots__ = ("w", "leaf", "l", "r", "aL0", "aR0", "t")

    def __init__(self, w=0, leaf=False, l=None, r=None):
        self.leaf = leaf; self.l = l; self.r = r; self.t = 1
        if leaf:
            self.w = w
        else:
            g = gcd(l.w, r.w)
            self.aL0 = r.w // g; self.aR0 = l.w // g; self.w = l.w + r.w


def _sorted_split(ws):
    s = sorted(ws); n = len(s); o = []
    for i in range(n // 2):
        o += [s[i], s[n // 2 + i]]
    if n % 2:
        o.append(s[-1])
    return o


def _build_balanced(order):
    layer = [RNode(w, leaf=True) for w in order]
    while len(layer) > 1:
        nxt = [RNode(l=layer[i], r=layer[i + 1]) for i in range(0, len(layer) - 1, 2)]
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]


def _positions(root):
    """returns list of (x, d, leafnode) in pre-order."""
    out = []

    def place(n, x, d):
        if n.leaf:
            out.append((x, d, n)); return
        place(n.l, x - n.t * n.aL0, d + 1)
        place(n.r, x + n.t * n.aR0, d + 1)
    place(root, 0, 0)
    return out


def _under(node, target):
    if node is target:
        return True
    if node.leaf:
        return False
    return _under(node.l, target) or _under(node.r, target)


def _repair(root, Amax, maxiter=5000):
    for _ in range(maxiter):
        pos = _positions(root)
        seen = {}
        clash = None
        for (x, d, n) in pos:
            if (x, d) in seen:
                clash = (seen[(x, d)], n); break
            seen[(x, d)] = n
        if clash is None:
            return True
        a, b = clash
        # walk to the lowest rod that separates leaf a and leaf b
        cur = root
        lca = None
        while not cur.leaf:
            a_left = _under(cur.l, a); b_left = _under(cur.l, b)
            if a_left and b_left:
                cur = cur.l
            elif (not a_left) and (not b_left):
                cur = cur.r
            else:
                lca = cur; break
        if lca is None:
            return False
        bumped = False
        for cand in (lca, lca.l, lca.r):
            if cand.leaf:
                continue
            if cand.t * max(cand.aL0, cand.aR0) < Amax:
                cand.t += 1; bumped = True; break
        if not bumped:
            return False
    return False


def reference_positions(ws, Amax):
    root = _build_balanced(_sorted_split(ws))
    _repair(root, Amax)
    return [(x, d) for (x, d, _n) in _positions(root)]


# ---------------------------------------------------------------- parse + validate
def main():
    try:
        N, Amax, ws = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not toks:
        fail("empty output")

    idx = [0]

    def nxt():
        if idx[0] >= len(toks):
            fail("truncated output")
        t = toks[idx[0]]; idx[0] += 1
        return t

    def nxt_int():
        t = nxt()
        try:
            return int(t)
        except Exception:
            fail("non-integer token '%s'" % t)

    M = nxt_int()
    if M != 2 * N - 1:
        fail("M must equal 2N-1 = %d" % (2 * N - 1))

    nodes = []
    for i in range(M):
        typ = nxt()
        if typ == "L":
            nodes.append(("L", nxt_int()))
        elif typ == "I":
            l = nxt_int(); r = nxt_int(); aL = nxt_int(); aR = nxt_int()
            nodes.append(("I", l, r, aL, aR))
        else:
            fail("bad node type '%s'" % typ)

    claimed = [False] * M
    for nd in nodes:
        if nd[0] == "I":
            _, l, r, aL, aR = nd
            if l == r:
                fail("a rod points both children at the same node")
            for c in (l, r):
                if c < 0 or c >= M:
                    fail("child id %d out of range" % c)
                if claimed[c]:
                    fail("node %d claimed by two rods" % c)
                claimed[c] = True
    roots = [i for i in range(M) if not claimed[i]]
    if len(roots) != 1:
        fail("tree must have exactly one root (found %d)" % len(roots))
    root = roots[0]

    sys.setrecursionlimit(100000)
    visited = [False] * M
    leaves_w = []

    def subtree(u):
        if visited[u]:
            fail("cycle detected")
        visited[u] = True
        nd = nodes[u]
        if nd[0] == "L":
            leaves_w.append(nd[1]); return nd[1]
        _, l, r, aL, aR = nd
        wl = subtree(l); wr = subtree(r)
        if not (1 <= aL <= Amax and 1 <= aR <= Amax):
            fail("arm out of range [1,%d] at node %d" % (Amax, u))
        if wl * aL != wr * aR:
            fail("torque imbalance at node %d: %d*%d != %d*%d" % (u, wl, aL, wr, aR))
        return wl + wr

    subtree(root)
    if not all(visited):
        fail("some nodes are not reachable from the root")
    if sorted(leaves_w) != sorted(ws):
        fail("leaf weights are not the required multiset")

    leaf_pos = []

    def place(u, x, d):
        nd = nodes[u]
        if nd[0] == "L":
            leaf_pos.append((x, d)); return
        _, l, r, aL, aR = nd
        place(l, x - aL, d + 1)
        place(r, x + aR, d + 1)

    place(root, 0, 0)
    seen = set()
    for p in leaf_pos:
        if p in seen:
            fail("two leaves collide at (x=%d, depth=%d)" % p)
        seen.add(p)

    F = objective(leaf_pos, N, Amax)
    B = max(FLOOR, objective(reference_positions(ws, Amax), N, Amax))
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
