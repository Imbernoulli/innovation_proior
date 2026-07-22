# TIER: greedy
# MODE SELECTION (the sophisticated textbook answer): compute the cost of pure
# forward mode and pure reverse mode, and output whichever is cheaper.  This is the
# "it's just a mode choice" reflex -- exactly what the innovation hook says is NOT
# enough.  It ignores the graph-cut structure, so it stays far from cross-country.
import sys


def read_graph():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    V = int(next(it)); E = int(next(it)); M = int(next(it)); N = int(next(it))
    outadj = {u: {} for u in range(V)}
    inadj = {v: set() for v in range(V)}
    for _ in range(E):
        u = int(next(it)); v = int(next(it)); next(it)  # value irrelevant to op count
        outadj[u][v] = 1
        inadj[v].add(u)
    intermediates = list(range(M, V - N))
    return V, intermediates, outadj, inadj


def cost_of(order, V, outadj0, inadj0):
    out = {u: dict(d) for u, d in outadj0.items()}
    inn = {v: set(s) for v, s in inadj0.items()}
    ops = 0
    for v in order:
        preds = list(inn[v]); succs = list(out[v].keys())
        ops += len(preds) * len(succs)
        for p in preds:
            op = out[p]
            for s in succs:
                if s not in op:
                    op[s] = 1
                    inn[s].add(p)
        for p in preds:
            del out[p][v]
        for s in succs:
            inn[s].discard(v)
        out[v] = {}; inn[v] = set()
    return ops


def main():
    V, inter, outadj, inadj = read_graph()
    fwd = sorted(inter)
    rev = sorted(inter, reverse=True)
    cf = cost_of(fwd, V, outadj, inadj)
    cr = cost_of(rev, V, outadj, inadj)
    order = fwd if cf <= cr else rev
    sys.stdout.write(" ".join(map(str, order)) + "\n")


if __name__ == "__main__":
    main()
