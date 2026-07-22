import sys, random

# heat-corridor-sink-placement / "cooling a lava-tunnel server vault"
# Grid HxW. Outer border cells are INSULATING WALLS except the listed VENT cells
# (cold sinks, T=0). All interior cells are open conductive rock. Heat SOURCES inject
# integer wattage. The solver upgrades up to K interior cells to conductivity KHI.
# Deterministic: everything seeded from testId.

def build_case(i):
    rng = random.Random(50700 + i)
    KHI = 24
    if i == 1:     # easy: dominant source adjacent to a vent, weak secondary
        H, W = 24, 13; vents = [(0, 6), (23, 6)]
        sources = [(3, 6, 180), (16, 4, 55)]; K = 24
    elif i == 2:   # easy
        H, W = 26, 15; vents = [(0, 7), (25, 7)]
        sources = [(3, 7, 200), (22, 10, 60)]; K = 26
    elif i == 3:   # easy
        H, W = 30, 15; vents = [(0, 7), (29, 7)]
        sources = [(3, 7, 200), (26, 8, 70)]; K = 30
    elif i == 4:   # TRAP: deep central source far from every vent
        H, W = 34, 15; vents = [(0, 7), (33, 7)]
        sources = [(17, 7, 240), (6, 4, 110), (27, 11, 110)]; K = 34
    elif i == 5:   # TRAP
        H, W = 38, 17; vents = [(0, 8), (37, 8)]
        sources = [(19, 8, 260), (8, 5, 120), (30, 12, 120)]; K = 36
    elif i == 6:   # TRAP: single vent -> one long escape route
        H, W = 36, 17; vents = [(0, 8)]
        sources = [(24, 8, 240), (12, 5, 130)]; K = 34
    elif i == 7:   # TRAP: several deep sources
        H, W = 42, 17; vents = [(0, 8), (41, 8)]
        sources = [(21, 8, 270), (21, 13, 150), (10, 4, 120), (32, 12, 120)]; K = 40
    elif i == 8:   # TRAP
        H, W = 46, 19; vents = [(0, 9), (45, 9), (23, 0)]
        sources = [(23, 10, 280), (12, 6, 140), (34, 14, 130)]; K = 44
    elif i == 9:   # TRAP: tight budget, competing deep sources
        H, W = 44, 19; vents = [(0, 9), (43, 9)]
        sources = [(22, 9, 300), (22, 14, 170), (10, 5, 130), (34, 13, 130)]; K = 38
    else:          # i == 10, hardest
        H, W = 50, 19; vents = [(0, 9), (49, 9), (25, 0)]
        sources = [(25, 10, 320), (25, 14, 180), (12, 5, 150), (38, 14, 150)]; K = 48
    # small deterministic wattage jitter so the statement can't be pattern-matched
    sources = [(r, c, p + rng.randint(-6, 6)) for (r, c, p) in sources]
    return H, W, KHI, vents, sources, K


def main():
    i = int(sys.argv[1])
    H, W, KHI, vents, sources, K = build_case(i)
    out = []
    out.append("%d %d %d %d" % (H, W, KHI, K))
    out.append("%d" % len(sources))
    for (r, c, p) in sources:
        out.append("%d %d %d" % (r, c, p))
    out.append("%d" % len(vents))
    for (r, c) in vents:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
