# TIER: trivial
# Baseline mobile == the checker's reference construction: a complete balanced tree
# over a fixed leaf order with MINIMAL arms, then a deterministic collision-repair
# that bumps rod scales just enough to separate coincident leaves. All leaves land
# at one depth (zero depth-entropy) and hug the spine (low horizontal spread).
import sys
from math import gcd


class Node:
    __slots__ = ("w", "leaf", "l", "r", "aL0", "aR0", "t")

    def __init__(self, w=0, leaf=False, l=None, r=None):
        self.leaf = leaf; self.l = l; self.r = r; self.t = 1
        if leaf:
            self.w = w
        else:
            g = gcd(l.w, r.w)
            self.aL0 = r.w // g; self.aR0 = l.w // g; self.w = l.w + r.w


def sorted_split(ws):
    s = sorted(ws); n = len(s); o = []
    for i in range(n // 2):
        o += [s[i], s[n // 2 + i]]
    if n % 2:
        o.append(s[-1])
    return o


def build_balanced(order):
    layer = [Node(w, leaf=True) for w in order]
    while len(layer) > 1:
        nxt = [Node(l=layer[i], r=layer[i + 1]) for i in range(0, len(layer) - 1, 2)]
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]


def positions(root):
    out = []

    def place(n, x, d):
        if n.leaf:
            out.append((x, d, n)); return
        place(n.l, x - n.t * n.aL0, d + 1)
        place(n.r, x + n.t * n.aR0, d + 1)
    place(root, 0, 0)
    return out


def under(node, target):
    if node is target:
        return True
    if node.leaf:
        return False
    return under(node.l, target) or under(node.r, target)


def repair(root, Amax, maxiter=5000):
    for _ in range(maxiter):
        pos = positions(root); seen = {}; clash = None
        for (x, d, n) in pos:
            if (x, d) in seen:
                clash = (seen[(x, d)], n); break
            seen[(x, d)] = n
        if clash is None:
            return
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
            return
        for cand in (lca, lca.l, lca.r):
            if cand.leaf:
                continue
            if cand.t * max(cand.aL0, cand.aR0) < Amax:
                cand.t += 1; break
        else:
            return


def serialize(root):
    nodes = []

    def dfs(node):
        if node.leaf:
            nodes.append(("L", node.w))
            return len(nodes) - 1
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
    root = build_balanced(sorted_split(ws))
    repair(root, Amax)
    sys.stdout.write(serialize(root))


main()
