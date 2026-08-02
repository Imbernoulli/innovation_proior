# TIER: trivial
"""Reproduces the checker's own baseline: a breaker that (in practice) never
trips at all -- w_trip=T with k_trip=T requires the ENTIRE trace to be
failures before it would ever open, which none of these instances do. This
is exactly "always call, no breaker", i.e. the checker's reference B, so it
scores Ratio ~= 0.1 by construction."""
import sys


def main():
    head = sys.stdin.readline().split()
    T = int(head[0])
    sys.stdin.readline()  # outcomes, unused
    print(f"{T} {T} 1 1 1 0 1")


if __name__ == "__main__":
    main()
