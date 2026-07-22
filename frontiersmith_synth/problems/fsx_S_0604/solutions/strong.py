# TIER: strong
# CROSS-COUNTRY elimination via narrow-cut discovery.
#
# Insight: optimal Jacobian accumulation is NOT a forward/reverse mode choice -- it
# is a graph-cut problem.  The DAG funnels through narrow cuts; each cut is the
# cheap dimension to accumulate through.  We recover the layered cut structure
# (longest-path levels), then recursively split the layer chain AT ITS NARROWEST
# interior layer (divide-and-conquer matrix-chain association): fully collapse both
# sides down to that cut, then eliminate the (few) cut vertices last, when their
# in/out degrees are smallest.  This pre-accumulates around every bottleneck.
import sys
sys.setrecursionlimit(100000)


def read_graph():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    V = int(next(it)); E = int(next(it)); M = int(next(it)); N = int(next(it))
    preds = {v: [] for v in range(V)}
    for _ in range(E):
        u = int(next(it)); v = int(next(it)); next(it)
        preds[v].append(u)
    return V, M, N, preds


def main():
    V, M, N, preds = read_graph()
    # longest-path level (ids are topological, so a single ascending pass works)
    level = [0] * V
    for v in range(V):
        best = 0
        for u in preds[v]:
            if level[u] + 1 > best:
                best = level[u] + 1
        level[v] = best
    Lmax = max(level)
    # group vertices by level (0 = inputs, Lmax = outputs)
    bylevel = {}
    for v in range(V):
        bylevel.setdefault(level[v], []).append(v)
    width = {lv: len(vs) for lv, vs in bylevel.items()}

    order = []

    def emit(i, k):
        # eliminate every interior LEVEL strictly between i and k
        if k <= i + 1:
            return
        # narrowest interior level is the cheap cut to accumulate through
        best_j, best_w = None, None
        for j in range(i + 1, k):
            w = width.get(j, 0)
            if best_j is None or w < best_w:
                best_j, best_w = j, w
        emit(i, best_j)
        emit(best_j, k)
        order.extend(bylevel.get(best_j, []))

    emit(0, Lmax)

    # safety: append any interior vertex not yet covered (keeps output a valid perm)
    seen = set(order)
    for lv in range(1, Lmax):
        for v in bylevel.get(lv, []):
            if v not in seen:
                order.append(v); seen.add(v)

    sys.stdout.write(" ".join(map(str, order)) + "\n")


if __name__ == "__main__":
    main()
