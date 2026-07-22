# TIER: trivial
"""Spread the mass budget as evenly as possible across all beads and ignore
the target chord entirely.  This reproduces the checker's own baseline
construction (score ~0.1 by design)."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    n = int(data[p]); p += 1
    B = int(data[p]); p += 1
    # CAP, r, targets are read but unused by this tier
    p += 1  # CAP
    r = int(data[p]); p += 1
    p += 2 * r  # skip target pairs

    q, rem = divmod(B, n)
    e = [q + (1 if i < rem else 0) for i in range(n)]
    print(" ".join(map(str, e)))


if __name__ == "__main__":
    main()
