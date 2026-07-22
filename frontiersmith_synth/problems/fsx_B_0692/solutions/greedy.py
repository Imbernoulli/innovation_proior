# TIER: greedy
"""The obvious 'competent compiler' recipe: a Sethi-Ullman-style register
allocator with a spill heuristic. For every node used more than once it
compares two numbers -- (a) the cost of recomputing its whole subtree from
scratch on every use, vs (b) the cost of computing it once and paying
STORE+LOAD (memory, cost 8 each) for every reuse -- and picks whichever is
cheaper. This is "textbook register allocation": it reasons about *how
expensive a value is to rebuild*, but it has NO notion of *where on the
stack* a value currently sits. It never realizes that a value which is
about to be reused almost immediately could be grabbed with a single
DUP/SWAP/OVER/ROT (cost 1) instead of round-tripped through memory (cost
16) -- memory is the only reuse mechanism it knows, so it still massively
over-pays on every value it decides to spill but that a shuffle could have
served for a fraction of the cost."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    for _ in range(N):
        next(it)
    nodes = {}
    for i in range(1, M + 1):
        op = next(it); cl = next(it); cr = next(it)
        nodes[i] = (op, cl, cr)
    outs = [next(it) for _ in range(K)]

    sys.setrecursionlimit(1000000)

    usage = {}

    def bump(ref):
        usage[ref] = usage.get(ref, 0) + 1

    for i in range(1, M + 1):
        _op, cl, cr = nodes[i]
        bump(cl); bump(cr)
    for o in outs:
        bump(o)

    size = {}
    for i in range(1, N + 1):
        size[("L", i)] = 1
    for i in range(1, M + 1):
        _op, cl, cr = nodes[i]
        clk = (cl[0], int(cl[1:])); crk = (cr[0], int(cr[1:]))
        size[("N", i)] = size[clk] + size[crk] + 1

    spill_nodes = set()
    for idx in range(1, M + 1):
        u = usage.get("N%d" % idx, 0)
        if u >= 2:
            s = size[("N", idx)]
            recompute_cost = s * u
            spill_cost = s + 9 + 8 * (u - 1)   # +1 DUP +8 STORE on creation, +8 per later LOAD
            if spill_cost < recompute_cost:
                spill_nodes.add(idx)

    lines = []
    computed = set()

    def visit(ref):
        if ref[0] == "L":
            lines.append("PUSH %d" % int(ref[1:]))
            return
        idx = int(ref[1:])
        if idx in computed:
            lines.append("LOAD %d" % idx)
            return
        op, cl, cr = nodes[idx]
        visit(cl)
        visit(cr)
        lines.append("OP %s" % op)
        if idx in spill_nodes:
            lines.append("DUP")
            lines.append("STORE %d" % idx)
            computed.add(idx)

    for o in outs:
        visit(o)
        lines.append("OUTPUT")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
