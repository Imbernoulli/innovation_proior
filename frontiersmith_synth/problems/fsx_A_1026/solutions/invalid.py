# TIER: invalid
"""Emits an out-of-range refractive-level grid (every cell above LMAX) -- must
score Ratio: 0.0 under strict feasibility checking."""
import sys


def main():
    toks = sys.stdin.read().split()
    W = int(toks[0]); H = int(toks[1])
    LMAX = int(toks[2])
    bad = LMAX + 5
    row = " ".join(str(bad) for _ in range(W))
    out = "\n".join(row for _ in range(H))
    print(out)


if __name__ == "__main__":
    main()
