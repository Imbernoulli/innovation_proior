# TIER: greedy
# Nearest-first router: repeatedly pick the still-pending operation whose two batches
# are currently closest, and route it. Reordering the schedule to serve nearby pairs
# first leaves batches in more useful places and beats the fixed-order baseline.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    rows = int(next(it)); cols = int(next(it)); K = int(next(it))
    V = rows * cols
    ops = [(int(next(it)), int(next(it))) for _ in range(K)]

    def dist(u, w):
        ru, cu = divmod(u, cols); rw, cw = divmod(w, cols)
        return abs(ru - rw) + abs(cu - cw)

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

    pos = list(range(V)); inv = list(range(V))
    pending = list(range(K))
    pending_set = set(pending)
    moves = []
    while pending_set:
        # pick min-distance pending op (tie-break by index)
        best = None; bestd = None
        for t in pending:
            if t not in pending_set:
                continue
            a, b = ops[t]
            d = dist(pos[a], pos[b])
            if bestd is None or d < bestd:
                bestd = d; best = t
        t = best
        a, b = ops[t]
        path = path_manhattan(pos[a], pos[b])
        for i in range(len(path) - 2):
            u, w = path[i], path[i + 1]
            tu, tw = inv[u], inv[w]
            inv[u], inv[w] = tw, tu
            pos[tu], pos[tw] = w, u
            moves.append("S %d %d" % (u, w))
        moves.append("X %d" % t)
        pending_set.discard(t)

    sys.stdout.write("%d\n%s\n" % (len(moves), "\n".join(moves)))

if __name__ == "__main__":
    main()
