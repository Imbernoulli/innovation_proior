import sys

# Format D checker -- reservoir-batch routing on a grid canal network.
#   1) Parse instance: grid (rows x cols), K mixing operations (batch pairs).
#      Batch i starts in reservoir i (identity placement).
#   2) Parse participant program of MOVES:
#         M
#         then M lines, each either
#            S u v   -- exchange the batches in directly-connected reservoirs u,v
#            X t     -- perform mixing operation t (its two batches must currently
#                       sit in directly-connected reservoirs); t used at most once.
#   3) FEASIBILITY gate (any violation -> Ratio: 0.0):
#         * every S must be on a canal (orthogonally adjacent reservoirs)
#         * every X's two batches must be adjacent at that moment; t in [0,K); no repeats
#         * ALL K operations must be performed
#         * strict token parsing (ints only) -> rejects nan/inf/garbage
#   4) Objective (minimize) = number of exchanges F.
#      Baseline B = exchanges used by the naive fixed-order router (below).
#      Ratio = min(1, 0.1 * B / F).

MAX_MOVES = 5_000_000

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    it = iter(inp)
    try:
        rows = int(next(it)); cols = int(next(it)); K = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= rows <= 40 and 2 <= cols <= 40 and 1 <= K <= 100000):
        fail("bad dims")
    V = rows * cols
    ops = []
    try:
        for _ in range(K):
            a = int(next(it)); b = int(next(it))
            if not (0 <= a < V and 0 <= b < V and a != b):
                fail("bad op")
            ops.append((a, b))
    except Exception:
        fail("bad ops")

    def adj(u, w):
        ru, cu = divmod(u, cols); rw, cw = divmod(w, cols)
        return abs(ru - rw) + abs(cu - cw) == 1

    # ---- naive baseline B: fixed list order, move batch a toward b along a
    #      canonical row-then-column shortest path ----
    def path_manhattan(src, dst):
        r0, c0 = divmod(src, cols); r1, c1 = divmod(dst, cols)
        path = [src]; r, c = r0, c0
        sr = 1 if r1 > r else -1
        while r != r1:
            r += sr; path.append(r * cols + c)
        sc = 1 if c1 > c else -1
        while c != c1:
            c += sc; path.append(r * cols + c)
        return path

    def naive_swaps():
        pos = list(range(V)); inv = list(range(V))
        total = 0
        for (a, b) in ops:
            path = path_manhattan(pos[a], pos[b])
            for i in range(len(path) - 2):
                u, w = path[i], path[i + 1]
                tu, tw = inv[u], inv[w]
                inv[u], inv[w] = tw, tu
                pos[tu], pos[tw] = w, u
                total += 1
        return total

    B = naive_swaps()

    # ---- parse & simulate participant ----
    out = open(sys.argv[2]).read().split()
    ot = iter(out)
    try:
        M = int(next(ot))
    except Exception:
        fail("no move count")
    if not (0 <= M <= MAX_MOVES):
        fail("bad move count")

    pos = list(range(V)); inv = list(range(V))
    done = [False] * K
    F = 0
    try:
        for _ in range(M):
            op = next(ot)
            if op == "S":
                u = int(next(ot)); w = int(next(ot))
                if not (0 <= u < V and 0 <= w < V):
                    fail("swap out of range")
                if not adj(u, w):
                    fail("swap not on a canal")
                tu, tw = inv[u], inv[w]
                inv[u], inv[w] = tw, tu
                pos[tu], pos[tw] = w, u
                F += 1
            elif op == "X":
                t = int(next(ot))
                if not (0 <= t < K):
                    fail("op index out of range")
                if done[t]:
                    fail("op performed twice")
                a, b = ops[t]
                if not adj(pos[a], pos[b]):
                    fail("op batches not adjacent")
                done[t] = True
            else:
                fail("bad move token")
    except StopIteration:
        fail("truncated move stream")

    if not all(done):
        fail("not all operations performed")

    if F == 0:
        # all operations already adjacent under the identity placement: perfect
        print("Ratio: 1.000000")
        return

    ratio = min(1.0, 0.1 * B / F)
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))

if __name__ == "__main__":
    main()
