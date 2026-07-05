# TIER: trivial
# First n non-reserved integers 0,1,2,... -- a near arithmetic progression.
# Reproduces the checker's internal baseline, so it scores ~0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    n, M, k = int(toks[0]), int(toks[1]), int(toks[2])
    forb = set(int(x) for x in toks[3:3 + k])
    A = []
    c = 0
    while len(A) < n and c <= M:
        if c not in forb:
            A.append(c)
        c += 1
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
