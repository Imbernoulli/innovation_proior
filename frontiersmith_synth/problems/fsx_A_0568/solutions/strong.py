# TIER: strong
# INSIGHT: a rod's balance wL*aL == wR*aR fixes only the RATIO aL:aR; the magnitude
# is a free integer scale (aL,aR) = (t*wR/g, t*wL/g). So statics never constrain the
# silhouette -- we spend the scale freedom to fling leaves outward into distinct
# horizontal bins, and pick a topology that (a) gives leaves a spread of depths and
# (b) groups high-gcd / equal-weight subtrees as siblings so t can run all the way to
# the arm ceiling. Several such constructions are built and the checker's own
# objective is used to keep the best feasible (in-range, collision-free) one.
import sys, math
from math import gcd
from collections import Counter


class Node:
    __slots__ = ("w", "leaf", "l", "r", "aL0", "aR0", "t")

    def __init__(self, w=0, leaf=False, l=None, r=None):
        self.leaf = leaf; self.l = l; self.r = r; self.t = 1
        if leaf:
            self.w = w
        else:
            g = gcd(l.w, r.w)
            self.aL0 = r.w // g; self.aR0 = l.w // g; self.w = l.w + r.w


def mk_int(a, b, Amax, mode):
    n = Node(l=a, r=b)
    tmax = max(1, Amax // max(n.aL0, n.aR0))
    n.t = tmax if mode == "max" else max(1, tmax // 2) if mode == "half" else 1
    return n


def positions(root):
    out = []

    def place(n, x, d):
        if n.leaf:
            out.append((x, d, n)); return
        place(n.l, x - n.t * n.aL0, d + 1)
        place(n.r, x + n.t * n.aR0, d + 1)
    place(root, 0, 0)
    return out


def internal_nodes(root):
    res = []

    def rec(n):
        if n.leaf:
            return
        res.append(n); rec(n.l); rec(n.r)
    rec(root)
    return res


def feasible(root, Amax):
    for n in internal_nodes(root):
        if not (1 <= n.t * n.aL0 <= Amax and 1 <= n.t * n.aR0 <= Amax):
            return False
    pos = positions(root)
    return len(set((x, d) for x, d, _ in pos)) == len(pos)


def objective(root, N, Amax):
    pos = positions(root)
    xs = [p[0] for p in pos]; ds = [p[1] for p in pos]
    binw = max(1, Amax // (2 * N)); lN = math.log(N)
    cb = Counter(x // binw for x in xs)
    Hx = -sum((c / N) * math.log(c / N) for c in cb.values()) / lN
    cd = Counter(ds)
    Hd = -sum((c / N) * math.log(c / N) for c in cd.values()) / lN
    return 0.6 * Hx + 0.4 * Hd


def under(node, target):
    if node is target:
        return True
    if node.leaf:
        return False
    return under(node.l, target) or under(node.r, target)


def repair(root, Amax, maxiter=4000):
    for _ in range(maxiter):
        pos = positions(root); seen = {}; clash = None
        for (x, d, n) in pos:
            if (x, d) in seen:
                clash = (seen[(x, d)], n); break
            seen[(x, d)] = n
        if clash is None:
            return True
        a, b = clash; cur = root; lca = None
        while not cur.leaf:
            al = under(cur.l, a); bl = under(cur.l, b)
            if al and bl:
                cur = cur.l
            elif (not al) and (not bl):
                cur = cur.r
            else:
                lca = cur; break
        if lca is None:
            return False
        for cand in (lca, lca.l, lca.r):
            if not cand.leaf and cand.t * max(cand.aL0, cand.aR0) < Amax:
                cand.t += 1; break
        else:
            return False
    return False


# ---- topologies ----
def caterpillar(order, Amax, mode, alt):
    sub = Node(order[0], leaf=True)
    for k in range(1, len(order)):
        lf = Node(order[k], leaf=True)
        if alt and (k % 2 == 0):
            sub = mk_int(lf, sub, Amax, mode)
        else:
            sub = mk_int(sub, lf, Amax, mode)
    return sub


def gcd_balanced(ws, Amax, mode):
    rem = [Node(w, leaf=True) for w in ws]
    while len(rem) > 1:
        best = None
        for i in range(len(rem)):
            for j in range(i + 1, len(rem)):
                g = gcd(rem[i].w, rem[j].w)
                if best is None or g > best[0]:
                    best = (g, i, j)
        _, i, j = best
        node = mk_int(rem[i], rem[j], Amax, mode)
        rem = [rem[k] for k in range(len(rem)) if k != i and k != j] + [node]
    return rem[0]


def serialize(root):
    nodes = []

    def dfs(node):
        if node.leaf:
            nodes.append(("L", node.w)); return len(nodes) - 1
        li = dfs(node.l); ri = dfs(node.r)
        nodes.append(("I", li, ri, node.t * node.aL0, node.t * node.aR0))
        return len(nodes) - 1

    dfs(root)
    out = [str(len(nodes))]
    for nd in nodes:
        out.append("L %d" % nd[1] if nd[0] == "L"
                   else "I %d %d %d %d" % (nd[1], nd[2], nd[3], nd[4]))
    return "\n".join(out) + "\n"


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); Amax = int(t[1]); ws = [int(x) for x in t[2:2 + N]]
    desc = sorted(ws, reverse=True)

    def builders():
        yield caterpillar(list(ws), Amax, "max", True)
        yield caterpillar(list(ws), Amax, "half", True)
        yield caterpillar(desc, Amax, "max", True)
        yield caterpillar(desc, Amax, "max", False)
        yield gcd_balanced(list(ws), Amax, "max")
        yield gcd_balanced(list(ws), Amax, "half")
        yield caterpillar(list(ws), Amax, "min", True)  # safe fallback

    best_root = None; best_val = -1.0
    for root in builders():
        ok = feasible(root, Amax)
        if not ok:
            ok = repair(root, Amax) and feasible(root, Amax)
        if not ok:
            continue
        v = objective(root, N, Amax)
        if v > best_val:
            best_val = v; best_root = root

    if best_root is None:  # ultimate fallback: minimal-arm caterpillar
        best_root = caterpillar(list(ws), Amax, "min", True)
    sys.stdout.write(serialize(best_root))


main()
