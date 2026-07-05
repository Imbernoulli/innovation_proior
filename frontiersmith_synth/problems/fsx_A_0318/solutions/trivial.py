# TIER: trivial
"""Fiber decomposition = the checker's baseline: one rank-1 term e_i (x) e_j (x) fiber
per nonzero mode-3 fiber.  R = number of nonzero fibers -> ratio ~ 0.1."""
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    terms = []
    for i in range(a):
        for j in range(b):
            if any(T[i][j][k] != 0 for k in range(c)):
                u = [0] * a; u[i] = 1
                v = [0] * b; v[j] = 1
                w = [T[i][j][k] for k in range(c)]
                terms.append((u, v, w))
    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(str(x) for x in u + v + w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
