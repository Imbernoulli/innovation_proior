import sys, math, random

# ------------------------------------------------------------------
# downwash-formation-morph  --  drone light-show morph generator (spoke model)
#
# The figure is built from S angular SPOKES.  Along each spoke sit L drones
# at radii r_0<r_1<...<r_{L-1} (spacing dR) and heights h_l (spacing dH).
# In the START figure F0 the outer drones are HIGH (radius and height both
# grow with l): a flared cone of struts.  In the TARGET figure F1 the SAME
# radii are used but the heights are REVERSED along each spoke (outer drones
# LOW): the cone inverts.
#
# Two natural ways to morph (targets are unlabeled):
#   * keep-RADIUS:  each drone stays on its own strut radius and moves purely
#     VERTICALLY to the reversed height.  Struts have distinct radii, so the
#     cones stay (almost) empty -- but the moves are long (dH >> dR).
#   * keep-HEIGHT:  each drone keeps its height and slides RADIALLY to the
#     mirror radius.  This is the Euclidean-nearest match (dR << dH, so it
#     looks cheapest) -- but two drones on a spoke swap radii and cross at
#     the same (x,y) at different heights: they pile into each other's
#     downwash cones.  This is the trap.
#
# The cone is a truncated cone (radius grows with vertical gap, capped at
# Hmax), so even the keep-radius mover shades an adjacent strut when it ends
# up far above it -- which is why a good solver also DESCENDS LAYER BY LAYER
# (highest target first, one wave per height band) to keep the cones empty.
# ------------------------------------------------------------------

def spoke_xy(r, ang):
    return int(round(r * math.cos(ang))), int(round(r * math.sin(ang)))

def build(testId):
    rng = random.Random(75860 + testId)

    if testId <= 2:
        S, L = 4, 3
    elif testId <= 4:
        S, L = 5, 4
    elif testId <= 6:
        S, L = 6, 4
    elif testId <= 8:
        S, L = 6, 5
    else:
        S, L = 8, 5
    N = S * L

    import os
    R0 = 260
    dR = int(os.environ.get("G_DR", "200"))   # radial strut spacing (clears cone at rest)
    trap_ids = {2, 3, 5, 6, 7, 8, 9, 10}
    if testId in trap_ids:
        dH = int(os.environ.get("G_DH", "240"))   # tall stack -> radial swap is the bait
    else:
        dH = int(os.environ.get("G_DHE", "150"))  # short stack -> nearest keeps radius
    H0 = 0

    # spoke angles, slightly jittered so the figure is not perfectly symmetric
    spoke_ang = []
    for s in range(S):
        a = 2 * math.pi * s / S + (rng.random() - 0.5) * (0.15 * 2 * math.pi / S)
        spoke_ang.append(a)

    F0, F1 = [], []
    for s in range(S):
        ang = spoke_ang[s]
        for l in range(L):
            r = R0 + l * dR
            xy = spoke_xy(r, ang)
            F0.append((xy[0], xy[1], H0 + l * dH))
            # target: same radius, reversed height along the spoke
            F1.append((xy[0], xy[1], H0 + (L - 1 - l) * dH))

    # shuffle the target cloud so the identity matching is meaningless
    order = list(range(N))
    rng.shuffle(order)
    F1 = [F1[j] for j in order]

    Rbase = int(os.environ.get("G_RBASE", "90"))
    Rslope_num, Rslope_den = int(os.environ.get("G_RSN", "1")), 10  # truncated-cone slope
    Hmax = (L - 1) * dH + dH              # cone reaches across the whole strut
    Wpen = int(os.environ.get("G_WPEN", "85000"))    # downwash weight (per cone-tick)
    Wmake = int(os.environ.get("G_WMAKE", "1000"))  # makespan weight (per wave used)
    W = L                                # available wave slots
    K = 6                                # sub-ticks per wave

    return (N, L, S, W, K, Rbase, Rslope_num, Rslope_den, Hmax, Wpen, Wmake, F0, F1)

def main():
    testId = int(sys.argv[1])
    (N, L, S, W, K, Rbase, Rsn, Rsd, Hmax, Wpen, Wmake, F0, F1) = build(testId)
    out = []
    # header keeps L and P=S so a solver can recover the strut structure
    out.append("%d %d %d %d %d" % (N, L, S, W, K))
    out.append("%d %d %d %d %d %d" % (Rbase, Rsn, Rsd, Hmax, Wpen, Wmake))
    for (x, y, z) in F0:
        out.append("%d %d %d" % (x, y, z))
    for (x, y, z) in F1:
        out.append("%d %d %d" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
