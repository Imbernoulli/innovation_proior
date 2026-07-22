# TIER: strong
# Insight: the sweep's patterns form a small DECISION TREE.  A handful of
# "discriminator" blocks -- whose zero/nonzero values differ across the family --
# identify which pattern you are in; once identity is known you multiply exactly
# that pattern's nonzero blocks and nothing else.  So build a shallow tree by
# repeatedly TESTing the block that best splits the surviving pattern set (the
# cheapest splitting invariant), and at each leaf multiply only that pattern's
# own blocks.  Worst-case cost = tree depth + M * (largest single pattern) --
# far below both "multiply the union" and "test every block".
import sys
sys.setrecursionlimit(1000000)

def main():
    data = sys.stdin.read().split('\n')
    B, P, M = map(int, data[0].split())
    nz_sets = []
    for p in range(P):
        row = data[1 + p]
        nz_sets.append(set(i for i, ch in enumerate(row) if ch == '1'))

    def union_of(S):
        u = set()
        for p in S:
            u |= nz_sets[p]
        return u

    # ---- build decision tree over pattern indices ----
    def build(S):
        if len(S) == 1:
            return ('leaf', sorted(nz_sets[S[0]]))
        cnt = {}
        for p in S:
            for b in nz_sets[p]:
                cnt[b] = cnt.get(b, 0) + 1
        bestkey = None; bestj = None; bestsplit = None
        for b, c in cnt.items():
            if c == len(S):
                continue  # nonzero in all -> does not split
            zero = [p for p in S if b not in nz_sets[p]]
            one = [p for p in S if b in nz_sets[p]]
            # prefer a balanced split, tie-break toward smaller child unions
            key = (max(len(zero), len(one)),
                   len(union_of(zero)) + len(union_of(one)), b)
            if bestkey is None or key < bestkey:
                bestkey = key; bestj = b; bestsplit = (zero, one)
        if bestj is None:
            return ('leaf', sorted(union_of(S)))
        zero, one = bestsplit
        return ('test', bestj, build(zero), build(one))

    tree = build(list(range(P)))

    # ---- serialize (leaves = M-chains, tests = T nodes); node 0 = root ----
    nodes = []
    def add(nd):
        nodes.append(nd); return len(nodes) - 1

    def emit(tr):
        if tr[0] == 'leaf':
            blocks = tr[1]
            cur = add(('H',))
            for blk in reversed(blocks):
                cur = add(('M', blk, cur))
            return cur
        else:
            _, j, z, o = tr
            zi = emit(z); oi = emit(o)
            return add(('T', j, zi, oi))

    root = emit(tree)

    # relabel so root = index 0 (discovery-order DFS)
    newid = {}; order = []
    def visit(u):
        newid[u] = len(order); order.append(u)
    visit(root)
    i = 0
    while i < len(order):
        u = order[i]; i += 1
        nd = nodes[u]
        if nd[0] == 'T':
            for c in (nd[2], nd[3]):
                if c not in newid:
                    visit(c)
        elif nd[0] == 'M':
            if nd[2] not in newid:
                visit(nd[2])
    out = []
    for u in order:
        nd = nodes[u]
        if nd[0] == 'H':
            out.append("H")
        elif nd[0] == 'T':
            out.append("T %d %d %d" % (nd[1], newid[nd[2]], newid[nd[3]]))
        else:
            out.append("M %d %d" % (nd[1], newid[nd[2]]))

    sys.stdout.write("%d\n" % len(out))
    sys.stdout.write("\n".join(out) + "\n")

main()
