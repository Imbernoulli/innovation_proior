# TIER: trivial
"""Mode-3 slab decomposition: one act per (stage i, timeslot j) pair, with the
genre profile equal to that fiber.  Rank = a*b  == the checker baseline."""
import sys


def main():
    toks = sys.stdin.read().split()
    a, b, c = int(toks[0]), int(toks[1]), int(toks[2])
    body = list(map(int, toks[3:3 + a * b * c]))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = body[idx]; idx += 1

    lines = []
    for i in range(a):
        for j in range(b):
            u = [0] * a; u[i] = 1
            v = [0] * b; v[j] = 1
            w = [T[i][j][k] for k in range(c)]
            lines.append(" ".join(map(str, u + v + w)))
    out = [str(a * b)] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
