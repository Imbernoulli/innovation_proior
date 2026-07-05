# TIER: trivial
# Naive fixed-order router: perform the operations in the given list order; for each,
# walk batch a toward batch b along a canonical row-then-column shortest path. This
# reproduces the checker's internal baseline exactly -> ratio ~= 0.1.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    rows = int(next(it)); cols = int(next(it)); K = int(next(it))
    V = rows * cols
    ops = [(int(next(it)), int(next(it))) for _ in range(K)]

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
    moves = []
    for t, (a, b) in enumerate(ops):
        path = path_manhattan(pos[a], pos[b])
        for i in range(len(path) - 2):
            u, w = path[i], path[i + 1]
            tu, tw = inv[u], inv[w]
            inv[u], inv[w] = tw, tu
            pos[tu], pos[tw] = w, u
            moves.append("S %d %d" % (u, w))
        moves.append("X %d" % t)

    sys.stdout.write("%d\n%s\n" % (len(moves), "\n".join(moves)))

if __name__ == "__main__":
    main()
