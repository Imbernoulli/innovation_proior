# TIER: trivial
"""Cut every fusion edge: ship each facet as its own separate 1x1 sticker.
Always feasible (no piece can ever self-overlap), but pays the per-piece
penalty n-1 times and this is exactly the checker's own baseline B."""
import sys


def main():
    data = sys.stdin.read().split()
    # n, m, penalty are the first three tokens; we do not even need them.
    print(0)
    print()


if __name__ == "__main__":
    main()
