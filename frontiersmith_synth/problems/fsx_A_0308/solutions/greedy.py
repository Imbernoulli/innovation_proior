# TIER: greedy
# Pick the CHEAPEST of the three fiber decompositions (fewest non-zero fibers over a mode).
#   fibers over k (harmonic)  -> ~B*L primitives  (this is the checker baseline orientation)
#   fibers over j (line)      -> ~B*H primitives
#   fibers over i (bus)       -> ~L*H primitives
# Emits a unit-vector (x) unit-vector (x) fiber for the winning orientation. Still a pure
# fiber decomposition -- it never exploits the low rank WITHIN a slice, so it stays well
# above the slice-rank strategy.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    B = int(next(it)); L = int(next(it)); H = int(next(it))
    T = [[[0] * H for _ in range(L)] for _ in range(B)]
    for k in range(H):
        for i in range(B):
            for j in range(L):
                T[i][j][k] = int(next(it))

    def e(n, x):
        return [1 if p == x else 0 for p in range(n)]

    def over_k():  # fiber over harmonic, indexed by (bus, line)
        ts = []
        for i in range(B):
            for j in range(L):
                fib = [T[i][j][k] for k in range(H)]
                if any(fib):
                    ts.append((e(B, i), e(L, j), fib))
        return ts

    def over_j():  # fiber over line, indexed by (bus, harmonic)
        ts = []
        for i in range(B):
            for k in range(H):
                fib = [T[i][j][k] for j in range(L)]
                if any(fib):
                    ts.append((e(B, i), fib, e(H, k)))
        return ts

    def over_i():  # fiber over bus, indexed by (line, harmonic)
        ts = []
        for j in range(L):
            for k in range(H):
                fib = [T[i][j][k] for i in range(B)]
                if any(fib):
                    ts.append((fib, e(L, j), e(H, k)))
        return ts

    best = min([over_i(), over_j(), over_k()], key=len)

    out = [str(len(best))]
    for a, b, c in best:
        out.append(" ".join(map(str, a)))
        out.append(" ".join(map(str, b)))
        out.append(" ".join(map(str, c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
