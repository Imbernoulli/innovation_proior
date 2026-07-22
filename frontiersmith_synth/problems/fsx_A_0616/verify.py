import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Deterministic deposition simulator (the ground-truth self-modifying router).
# Given bedrock z0 and a T-step release schedule, returns the number of cells
# whose FINAL elevation is at or above the waterline (0.0).
#
#   * Between steps (before releases 1..T-1) every unit of ALREADY-DEPOSITED
#     sediment relaxes toward bedrock by factor (1-beta)  -- the sinking marsh.
#   * A release of mass M enters at the inlet (row sr, col 0) and is routed
#     downhill over the CURRENT terrain (cells processed by descending
#     elevation).  At a cell with in-transit mass q and steepest downhill drop
#     smax, a fraction  dep = d0 + (1-d0)/(1 + kslope*smax)  settles there and
#     the remainder is split among strictly-lower 4-neighbours in proportion to
#     their drop.  A pit deposits everything; the open sea edge (col W-1) keeps
#     the settled fraction and loses the rest to deep water.
#   * Deposits raise terrain and therefore reroute every LATER release.
# ---------------------------------------------------------------------------
def simulate(zbed, H, W, sr, schedule, d0, kslope, beta):
    N = H * W
    z = [zbed[r][c] for r in range(H) for c in range(W)]   # flat working grid
    bed = z[:]
    T = len(schedule)
    src = sr * W + 0
    neigh = [None] * N
    for r in range(H):
        for c in range(W):
            i = r * W + c
            lst = []
            if c + 1 < W: lst.append(i + 1)
            if c - 1 >= 0: lst.append(i - 1)
            if r + 1 < H: lst.append(i + W)
            if r - 1 >= 0: lst.append(i - W)
            neigh[i] = lst
    for t in range(T):
        if t > 0 and beta > 0.0:
            keep = 1.0 - beta
            for i in range(N):
                d = z[i] - bed[i]
                if d != 0.0:
                    z[i] = bed[i] + d * keep
        M = schedule[t]
        if M <= 0.0:
            continue
        order = sorted(range(N), key=lambda i: (-z[i], i))
        flux = [0.0] * N
        flux[src] += M
        dz = [0.0] * N
        for i in order:
            q = flux[i]
            if q <= 1e-15:
                continue
            zi = z[i]
            drops = []
            tot = 0.0
            for nb in neigh[i]:
                d = zi - z[nb]
                if d > 0.0:
                    drops.append((nb, d))
                    tot += d
            smax = 0.0
            for _, d in drops:
                if d > smax:
                    smax = d
            dep_frac = d0 + (1.0 - d0) / (1.0 + kslope * smax)
            if (i % W) == W - 1:               # open sea edge: settle, lose rest
                dz[i] += q * dep_frac
                continue
            if not drops:                      # interior pit: everything settles
                dz[i] += q
                continue
            dep = q * dep_frac
            dz[i] += dep
            rem = q - dep
            for nb, d in drops:
                flux[nb] += rem * (d / tot)
        for i in range(N):
            if dz[i] > 0.0:
                z[i] += dz[i]
    return sum(1 for i in range(N) if z[i] >= 0.0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); T = int(next(it))
    Mtot = float(next(it)); d0 = float(next(it)); kslope = float(next(it))
    beta = float(next(it)); sr = int(next(it))
    z = [[float(next(it)) for _ in range(W)] for _ in range(H)]
    return H, W, T, Mtot, d0, kslope, beta, sr, z


def main():
    try:
        H, W, T, Mtot, d0, kslope, beta, sr, z = read_instance(sys.argv[1])
    except Exception:
        fail("bad input")

    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(raw) != T:
        fail("expected %d release values, got %d" % (T, len(raw)))
    sched = []
    for tok in raw:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric release %r" % tok)
        if not math.isfinite(v):
            fail("non-finite release")
        if v < -1e-9:
            fail("negative release")
        sched.append(max(0.0, v))
    if sum(sched) > Mtot * (1.0 + 1e-6) + 1e-9:
        fail("budget exceeded: sum=%.6f > Mtot=%.6f" % (sum(sched), Mtot))

    # participant objective
    F = simulate(z, H, W, sr, sched, d0, kslope, beta)

    # internal reference baseline B: dump the whole budget in the final release
    # (a feasible construction -- it channelises straight to the sea edge).
    base_sched = [0.0] * (T - 1) + [Mtot]
    B = simulate(z, H, W, sr, base_sched, d0, kslope, beta)
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
