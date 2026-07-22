# TIER: greedy
# The obvious approach: chain the leaves one at a time onto a spine and balance each
# new rod with MINIMAL integer arms (the smallest arms that satisfy wL*aL == wR*aR).
# This "just balance it" recipe never spends the free rod-scale, so the silhouette
# stays cramped -- far below what deliberate scaling can spread.
import sys
from math import gcd


class Node:
    __slots__ = ("w", "leaf", "l", "r", "aL", "aR")

    def __init__(self, w=0, leaf=False, l=None, r=None):
        self.leaf = leaf; self.l = l; self.r = r
        if leaf:
            self.w = w
        else:
            g = gcd(l.w, r.w)
            self.aL = r.w // g; self.aR = l.w // g; self.w = l.w + r.w


def serialize(root):
    nodes = []

    def dfs(node):
        if node.leaf:
            nodes.append(("L", node.w))
            return len(nodes) - 1
        li = dfs(node.l); ri = dfs(node.r)
        nodes.append(("I", li, ri, node.aL, node.aR))
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
    sub = Node(ws[0], leaf=True)
    for k in range(1, N):
        sub = Node(l=sub, r=Node(ws[k], leaf=True))  # subtree on left, new leaf on right
    sys.stdout.write(serialize(sub))


main()
