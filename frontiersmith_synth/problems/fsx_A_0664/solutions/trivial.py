# TIER: trivial
"""Naive identity encoding: codeword(u) = binary(u). Reproduces the checker's own
internal baseline construction exactly (ratio ~0.1 on every test)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    M = 1 << K
    B = N.bit_length() - 1
    # consume the transition table (unused by this solution)
    for _ in range(N * M):
        next(it)

    out = []
    for u in range(N):
        out.append(format(u, f"0{B}b"))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
