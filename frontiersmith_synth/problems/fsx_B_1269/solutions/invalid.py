# TIER: invalid
"""Deliberately infeasible: claims a direct source-to-target hop that never
exists as a treaty link in this family's layered network (every instance
routes through at least one intermediate jurisdiction), so the checker's
edge-existence check must reject it and score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    print(2)
    print(f"0 {n - 1}")


if __name__ == "__main__":
    main()
