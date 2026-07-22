import sys, random

# ---------------------------------------------------------------------------
# sediment-pulse-land-builder  --  instance generator
#   testId 1..10 is a difficulty ladder: grid, horizon T, subsidence beta and
#   the incised trap-channel all grow.  Everything is seeded from testId only.
# ---------------------------------------------------------------------------

def main():
    tid = int(sys.argv[1])
    rng = random.Random(46160 + 1009 * tid)

    # ---- difficulty ladder -------------------------------------------------
    if tid <= 3:            # small / gentle
        H = 11 + 2 * tid
        W = 14 + 2 * tid
        T = 6 + tid
        beta = 0.14 + 0.03 * tid
        vd = 0.6 + 0.15 * tid
    elif tid <= 7:          # medium
        H = 17 + (tid - 3)
        W = 20 + (tid - 3)
        T = 9 + ((tid - 3) // 2)
        beta = 0.22 + 0.03 * (tid - 3)
        vd = 1.0 + 0.12 * (tid - 3)
    else:                   # large / adversarial (deep channel, strong sinking)
        H = 23 + (tid - 7)
        W = 26 + (tid - 7)
        T = 11 + ((tid - 8) // 2)
        beta = 0.32 + 0.02 * (tid - 7)
        vd = 1.5 + 0.15 * (tid - 7)

    d0 = 0.03
    kslope = 9.0
    sr = H // 2

    # total sediment budget: scaled with the shoreline so a good schedule fills
    # a modest fraction of the marsh (headroom left above the reference)
    Mtot = round(3.2 * W + 0.9 * H, 3)

    # ---- bedrock bathymetry: sinking marsh, deepening offshore -------------
    base = 0.30
    off = 0.11                      # cross-shore deepening per cell
    z = [[-(base + off * c) + rng.uniform(-0.05, 0.05) for c in range(W)]
         for _ in range(H)]

    # incised trap channel straight offshore from the inlet row (the plug/scour
    # that channelises an un-shaped flood straight out to the deep-water edge)
    for c in range(W):
        z[sr][c] -= vd
    # a couple of shallow along-shore rises (planted lateral targets)
    for _ in range(2):
        rr = rng.randrange(1, H - 1)
        if abs(rr - sr) <= 1:
            continue
        for c in range(W):
            z[rr][c] += 0.04 * rng.uniform(0.5, 1.5)

    # ---- emit --------------------------------------------------------------
    out = []
    out.append("%d %d %d" % (H, W, T))
    out.append("%.6f %.6f %.6f %.6f %d" % (Mtot, d0, kslope, beta, sr))
    for r in range(H):
        out.append(" ".join("%.6f" % z[r][c] for c in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
