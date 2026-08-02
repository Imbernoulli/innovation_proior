# TIER: trivial
"""Reproduces the checker's own internal baseline: the fixed "backbone" route
shipped at the end of the input (an always-compliant, real/high-substance-
only route through the FIRST candidate jurisdiction of every layer). Makes
no attempt to search the treaty network for anything better."""
import sys


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    n = int(toks[ptr]); ptr += 1
    m = int(toks[ptr]); ptr += 1
    ptr += 5           # V0 baseline_rate_bp gamma T_min T_max
    ptr += n            # substance scores
    ptr += 5 * m         # edges
    b = int(toks[ptr]); ptr += 1
    backbone = [int(toks[ptr + i]) for i in range(b)]
    print(b)
    print(" ".join(str(x) for x in backbone))


if __name__ == "__main__":
    main()
