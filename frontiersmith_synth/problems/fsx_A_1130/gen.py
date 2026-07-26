import sys, random, math

# ---- fixed world constants (kept constant across tests so the baseline reach is stable) ----
VMAX = 3
SINK = 3
AMAX = 90
A0 = 48
COORD = 45          # start/thermal/beacon coordinates roughly in [-COORD, COORD]

TRAP_IDS = {4, 6, 8, 9, 10}   # >=3 of the 10 cases engineer a crowd trap


def clampi(v, lo, hi):
    return max(lo, min(hi, int(v)))


def main():
    tid = int(sys.argv[1])
    rng = random.Random(9137 + 17 * tid)

    # ---- difficulty ladder ----
    if tid <= 3:
        G = 5 + tid          # 6,7,8
        K = 3
        M = 6 + tid           # 7,8,9
        T = 26 + 2 * tid
    elif tid <= 6:
        G = 8 + (tid - 3)     # 9,10,11
        K = 4 + (tid - 3) % 2
        M = 9 + (tid - 3)
        T = 32 + 2 * (tid - 3)
    else:
        G = 10 + (tid - 6)    # 11..14 -> capped below
        K = 5 + (tid - 6) % 3
        M = 11 + (tid - 6)
        T = 36 + 2 * (tid - 6)

    G = clampi(G, 5, 13)
    K = clampi(K, 3, 8)
    M = clampi(M, 6, 15)
    T = clampi(T, 24, 48)

    trap = tid in TRAP_IDS

    gliders = []
    thermals = []
    beacons = []

    # ---- thermals ----
    dom_c = None
    if trap:
        # one dominant thermal (looks unbeatable on paper: far richer than anything else)
        dom_c = (rng.randint(-COORD // 2, COORD // 2), rng.randint(-COORD // 2, COORD // 2))
        dom_v = (rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1]))
        dom_R = rng.randint(5, 7)
        dom_L = rng.randint(85, 115)
        thermals.append((dom_c[0], dom_c[1], dom_v[0], dom_v[1], dom_R, dom_L))
        for k in range(1, K):
            cx = rng.randint(-COORD, COORD)
            cy = rng.randint(-COORD, COORD)
            vx = rng.choice([-1, 0, 1])
            vy = rng.choice([-1, 0, 1])
            R = rng.randint(4, 7)
            L = rng.randint(18, 34)   # modest but genuinely usable alone (net positive vs sink)
            thermals.append((cx, cy, vx, vy, R, L))
    else:
        for k in range(K):
            cx = rng.randint(-COORD, COORD)
            cy = rng.randint(-COORD, COORD)
            vx = rng.choice([-1, 0, 1])
            vy = rng.choice([-1, 0, 1])
            R = rng.randint(4, 7)
            L = rng.randint(20, 55)
            thermals.append((cx, cy, vx, vy, R, L))
    rng.shuffle(thermals)

    # ---- glider starts ----
    if trap:
        # cluster most gliders a short, genuinely-reachable hop from the dominant thermal
        # (not sitting on top of it) so it is BOTH the richest AND the nearest usable
        # thermal for the pack -- an oblivious pursuit collides in time as well as space.
        ang0 = rng.uniform(0, 6.283185307)
        n_hub = max(3, int(G * 0.7))
        for i in range(G):
            if i < n_hub:
                ang = ang0 + rng.uniform(-1.0, 1.0)
                r = rng.uniform(10, 18)
                sx = dom_c[0] + round(r * math.cos(ang))
                sy = dom_c[1] + round(r * math.sin(ang))
            else:
                sx = rng.randint(-COORD, COORD)
                sy = rng.randint(-COORD, COORD)
            gliders.append((clampi(sx, -60, 60), clampi(sy, -60, 60)))
    else:
        for i in range(G):
            gliders.append((rng.randint(-COORD, COORD), rng.randint(-COORD, COORD)))

    # ---- beacons ----
    # kept within roughly the fleet's max-climb reach so most (not all) beacons are
    # genuinely competitive targets -- a fully-scattered field just adds unreachable noise.
    BSPREAD = round(1.5 * COORD)
    for m in range(M):
        bx = rng.randint(-BSPREAD, BSPREAD)
        by = rng.randint(-BSPREAD, BSPREAD)
        Rb = rng.randint(2, 4)
        val = rng.randint(6, 40)
        beacons.append((bx, by, Rb, val))

    # guarantee a positive checker baseline: a modest "gimme" beacon within straight-line
    # (no-thermal) reach of glider 0, so B > 0 on every test.
    reach = VMAX * (A0 // SINK) - 4
    gx, gy = gliders[0]
    ang = rng.uniform(0, 6.283185307)
    gimme_x = clampi(gx + round(reach * 0.6 * math.cos(ang)), -BSPREAD, BSPREAD)
    gimme_y = clampi(gy + round(reach * 0.6 * math.sin(ang)), -BSPREAD, BSPREAD)
    beacons[0] = (gimme_x, gimme_y, 3, rng.randint(6, 12))

    out = []
    out.append(f"{G} {K} {M} {T} {VMAX} {SINK} {AMAX} {A0}")
    for (sx, sy) in gliders:
        out.append(f"{sx} {sy}")
    for (cx, cy, vx, vy, R, L) in thermals:
        out.append(f"{cx} {cy} {vx} {vy} {R} {L}")
    for (bx, by, Rb, val) in beacons:
        out.append(f"{bx} {by} {Rb} {val}")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
