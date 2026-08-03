import sys

# Deterministic instance generator for "Quantum Lab Wiring: Shielded Spool Placement".
# `python3 gen.py <testId>` prints ONE instance. Seeded by testId only.
#
# Difficulty ladder (medium scale, variant #7): the pad budget N and the number of
# forbidden zones K both grow with the test id. Zones are always fully inside the
# central box [0.15, 0.85]^2, keeping the outer margin of the floor clear (so the
# baseline bottom-row construction is always feasible).

LADDER = [
    (8, 2), (10, 2), (12, 3), (15, 3), (18, 4),
    (22, 4), (26, 5), (30, 5), (35, 6), (40, 7),
]

S = 1.0
LO, HI = 0.15, 0.85          # forbidden zones live strictly inside this box
GMIN, GMAX = 0.03, 0.10      # zone radius range


def lcg(seed):
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield x / float(0x7FFFFFFF)


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    N, K = LADDER[idx]

    rng = lcg(9973 * (idx + 1) + 41)
    zones = []
    for _ in range(K):
        g = GMIN + (GMAX - GMIN) * next(rng)
        cx = (LO + g) + (HI - LO - 2 * g) * next(rng)
        cy = (LO + g) + (HI - LO - 2 * g) * next(rng)
        zones.append((cx, cy, g))

    out = ["%d %.1f %d" % (N, S, K)]
    for (cx, cy, g) in zones:
        out.append("%.6f %.6f %.6f" % (cx, cy, g))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
