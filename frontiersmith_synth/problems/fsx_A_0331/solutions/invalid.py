# TIER: invalid
"""Emit a deliberate resonance triple: the all-zero signature plus e0 and 2*e0, whose
coordinatewise sum is 0 mod 3. This is collinear -> must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    z = ["0"] * n
    a = ["0"] * n; a[0] = "1"
    b = ["0"] * n; b[0] = "2"
    print(" ".join(z))
    print(" ".join(a))
    print(" ".join(b))


if __name__ == "__main__":
    main()
