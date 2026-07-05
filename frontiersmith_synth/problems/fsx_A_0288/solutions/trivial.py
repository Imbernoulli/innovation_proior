# TIER: trivial
# Baseline: one rank-1 gadget per non-zero frontal (mode-3) fiber.
#   term = e_i (x) e_j (x) fiber_k   -> R = number of non-zero (i,j) fibers = checker baseline B.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for k in range(K):
        for i in range(I):
            for j in range(J):
                T[i][j][k] = int(next(it))

    terms = []
    for i in range(I):
        for j in range(J):
            fib = [T[i][j][k] for k in range(K)]
            if any(v != 0 for v in fib):
                a = [1 if x == i else 0 for x in range(I)]
                b = [1 if x == j else 0 for x in range(J)]
                terms.append((a, b, fib))

    out = [str(len(terms))]
    for a, b, c in terms:
        out.append(" ".join(map(str, a)))
        out.append(" ".join(map(str, b)))
        out.append(" ".join(map(str, c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
