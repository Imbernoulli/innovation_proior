import sys

# testId -> (p, d):  p = prime field size, d = degree of each input polynomial.
# The ladder mixes "large field" cases (p >= 2d+1: enough evaluation points exist,
# so an interpolation scheme reaches the 2d+1 rank bound) with "small field" traps
# (p < 2d+1: NOT enough points, so a naive point-evaluation coder is stuck and must
# fall back; only a CRT-over-F_p[x] scheme with irreducible moduli stays cheap).
CASES = {
    1: (7, 3),
    2: (7, 4),
    3: (11, 4),
    4: (7, 5),
    5: (11, 5),
    6: (7, 6),
    7: (13, 6),
    8: (7, 7),
    9: (11, 7),
    10: (17, 7),
}


def main():
    t = int(sys.argv[1])
    if t not in CASES:
        # clamp into range so the harness never crashes on out-of-range ids
        t = ((t - 1) % 10) + 1
    p, d = CASES[t]
    sys.stdout.write("%d %d\n" % (p, d))


main()
