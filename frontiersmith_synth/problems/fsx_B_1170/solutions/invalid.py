# TIER: invalid
"""Emits an out-of-range guess (source placed ON the boundary, which is
never a legal interior source cell) -- must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N, T, K = int(toks[0]), int(toks[1]), int(toks[2])
    out = []
    for _ in range(K):
        out.append(f"0 0 {T + 5}")  # boundary cell AND out-of-range onset
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
