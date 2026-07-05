# TIER: trivial
"""Diagonal configuration {0, e_1, ..., e_n} -- reproduces the checker baseline B.
Size = n + 1, so it scores ~0.1."""
import sys


def main():
    n = int(sys.stdin.readline().split()[0])
    out = []
    out.append(" ".join(["0"] * n))          # origin
    for i in range(n):
        v = ["0"] * n
        v[i] = "1"
        out.append(" ".join(v))              # e_i
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
