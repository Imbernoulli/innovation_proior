import sys
import math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad input file")
    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    # ---------- parse instance ----------
    try:
        it = iter(inp)
        n = int(next(it))
        s_budget = int(next(it))
        size = [0] * (n + 1)
        w = [0.0] * (n + 1)
        parent = [0] * (n + 1)
        up = [0] * (n + 1)
        down = [0] * (n + 1)
        size[1] = int(next(it))
        w[1] = float(next(it))
        children = [[] for _ in range(n + 1)]
        for i in range(2, n + 1):
            parent[i] = int(next(it))
            up[i] = int(next(it))
            down[i] = int(next(it))
            size[i] = int(next(it))
            w[i] = float(next(it))
            children[parent[i]].append(i)
    except Exception:
        fail("malformed input")

    if n < 1 or s_budget < size[1]:
        fail("degenerate instance")

    # ---------- internal baseline B: single exemplar at root, everyone ascends ----------
    cost_to_root = [0.0] * (n + 1)
    order = sorted(range(2, n + 1))  # parent[i] < i by construction
    for i in order:
        cost_to_root[i] = cost_to_root[parent[i]] + up[i]
    B = sum(w[i] * cost_to_root[i] for i in range(2, n + 1))
    B = max(1e-9, B)

    # ---------- parse participant output ----------
    def rd_int(it):
        return int(next(it))

    try:
        ot = iter(out_tokens)
        k = rd_int(ot)
    except Exception:
        fail("missing checkpoint count")
    if k < 0 or k > n:
        fail("bad checkpoint count")

    checkpoints = []
    is_ckpt = [False] * (n + 1)
    try:
        for _ in range(k):
            c = rd_int(ot)
            if c < 1 or c > n:
                fail("checkpoint id out of range")
            if is_ckpt[c]:
                fail("duplicate checkpoint")
            is_ckpt[c] = True
            checkpoints.append(c)
    except Exception:
        fail("truncated checkpoint list")

    if k < 1:
        fail("must keep at least one exemplar")

    total_size = sum(size[c] for c in checkpoints)
    if total_size > s_budget:
        fail("checkpoint budget exceeded: %d > %d" % (total_size, s_budget))

    # ---------- parse pointer pairs ----------
    ptr = [None] * (n + 1)
    covered = [False] * (n + 1)
    n_pairs = n - k
    try:
        for _ in range(n_pairs):
            v = rd_int(ot)
            t = rd_int(ot)
            if v < 1 or v > n or t < 1 or t > n:
                fail("pointer id out of range")
            if is_ckpt[v]:
                fail("checkpoint %d must not also carry a pointer" % v)
            if covered[v]:
                fail("duplicate pointer entry for %d" % v)
            adjacent = (t == parent[v]) or (v == parent[t])
            if not adjacent:
                fail("pointer %d -> %d is not a tree edge" % (v, t))
            ptr[v] = t
            covered[v] = True
    except SystemExit:
        raise
    except Exception:
        fail("truncated pointer list")

    # no leftover tokens
    try:
        next(ot)
        fail("trailing garbage in output")
    except StopIteration:
        pass

    for v in range(1, n + 1):
        if not is_ckpt[v] and not covered[v]:
            fail("no pointer registered for %d" % v)

    # ---------- resolve chains (memoized, cycle-safe) ----------
    UNVISITED, IN_PROGRESS, DONE = 0, 1, 2
    state = [UNVISITED] * (n + 1)
    chain_cost = [0.0] * (n + 1)
    for c in checkpoints:
        state[c] = DONE
        chain_cost[c] = 0.0

    def hop_cost(v, t):
        if t == parent[v]:
            return up[v]
        else:  # v == parent[t]
            return down[t]

    for start in range(1, n + 1):
        if state[start] == DONE:
            continue
        path = []
        v = start
        while state[v] == UNVISITED:
            state[v] = IN_PROGRESS
            path.append(v)
            v = ptr[v]
            if state[v] == IN_PROGRESS:
                fail("pointer cycle detected involving %d" % v)
        # v is now DONE (a checkpoint or previously resolved node)
        base = chain_cost[v]
        for node in reversed(path):
            base = base + hop_cost(node, ptr[node])
            chain_cost[node] = base
            state[node] = DONE

    F = sum(w[i] * chain_cost[i] for i in range(1, n + 1) if not is_ckpt[i])

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    if not math.isfinite(sc):
        fail("non-finite score")
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
