# TIER: greedy
# Probe-all: the obvious adaptive recipe.  TEST every block in turn; whenever a
# block comes back nonzero, MULTIPLY it, then move on.  Correct for every
# pattern, but the worst-case cost pays B tests on EVERY pattern -- it never
# exploits that the sweep's patterns share structure, so it drowns in test cost.
import sys
sys.setrecursionlimit(1000000)

def relabel(nodes, root):
    newid = {}
    order = []
    stack = [root]
    seen = set()
    # iterative DFS assigning ids in discovery order (root = 0)
    def visit(u):
        newid[u] = len(order)
        order.append(u)
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
            c = nd[2]
            if c not in newid:
                visit(c)
    out = []
    for u in order:
        nd = nodes[u]
        if nd[0] == 'H':
            out.append("H")
        elif nd[0] == 'T':
            out.append("T %d %d %d" % (nd[1], newid[nd[2]], newid[nd[3]]))
        else:
            out.append("M %d %d" % (nd[1], newid[nd[2]]))
    return out

def main():
    data = sys.stdin.read().split('\n')
    B, P, M = map(int, data[0].split())
    nodes = []
    def add(nd):
        nodes.append(nd); return len(nodes) - 1
    halt = add(('H',))
    cont = halt
    for j in reversed(range(B)):
        mnode = add(('M', j, cont))
        tnode = add(('T', j, cont, mnode))  # zero -> skip; nonzero -> multiply then continue
        cont = tnode
    out = relabel(nodes, cont)
    sys.stdout.write("%d\n" % len(out))
    sys.stdout.write("\n".join(out) + "\n")

main()
