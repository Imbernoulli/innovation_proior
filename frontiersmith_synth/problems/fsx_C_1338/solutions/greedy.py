# TIER: greedy
"""The obvious recipe: read the adhesive library, find the type with the
LARGEST shear stiffness (which is also individually the strongest type in
this library -- "maximize stiffness maximizes joint strength"), and use it
uniformly on every segment. This is individually rational -- in the
sub-critical size regime it genuinely IS the best single uniform choice --
but it never models the interfacial-stress-concentration mechanism at all,
so it walks straight into the edge-stress collapse once the bond line is
long enough."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    next(it); next(it)              # Csub dAlpha
    C = int(next(it))
    for _ in range(C):
        next(it)                    # dT_1..dT_C
    lib = []
    for _ in range(M):
        k = float(next(it)); s = float(next(it))
        lib.append((k, s))

    j_star = max(range(M), key=lambda j: lib[j][0])
    print(" ".join([str(j_star)] * N))


if __name__ == "__main__":
    main()
