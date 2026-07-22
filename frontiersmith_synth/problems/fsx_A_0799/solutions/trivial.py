# TIER: trivial
"""Baseline: spread the stiffness budget perfectly UNIFORMLY across all segments.
This exactly reproduces the checker's internal baseline B by construction, so it
scores ~0.1 -- and it is also the exact spectrum the drum corps' comb was planted
on, so it is the "obvious", resonance-colliding do-nothing move."""
import sys


def main():
    data = sys.stdin.read().split()
    S = int(data[0]); BUDGET = int(data[1])
    base = BUDGET // S
    rem = BUDGET - base * S
    ks = [base + (1 if i < rem else 0) for i in range(S)]
    print(" ".join(map(str, ks)))


if __name__ == "__main__":
    main()
