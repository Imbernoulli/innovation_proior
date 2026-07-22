import sys, math

# ---------------------------------------------------------------------------
# Checker for single-track-meetpass-rhythm (heterogeneous speeds).
#
# Instance (stdin of the solver):
#   line1: S TMAX
#   line2: cap[0..S-1]
#   line3: N
#   next N: dir r d w v h     (dir 0=east 0->S-1, 1=west S-1->0; h = ticks/block)
#
# Participant artifact (stdout): N lines, each S-1 integers = the tick train j
# ENTERS each block along its route (route order).  arr_j = e[S-2] + h_j.
#
# Feasibility (any violation -> Ratio: 0.0):
#   * exactly N lines of S-1 integers in [0, TMAX]
#   * e[0] >= r_j  and  e[k+1] >= e[k] + h_j
#   * block exclusivity: on each physical block, the occupancy windows
#     [e_j, e_j+h_j) of the trains using it never overlap (touching ok)
#     -> forbids head-on meets AND in-block overtakes
#   * siding capacity: at each intermediate station i, at most cap[i] trains
#     dwell simultaneously
#
# Objective (minimize):
#   F = sum_j w_j*(arr_j - r_j) + sum_j v_j*max(0, arr_j - d_j)
#
# Baseline B: fully serialized dispatch in release order (each train departs
# only once the previous one has completely cleared the line) -- always
# feasible.  Minimization normalization:  Ratio = min(1, 0.1 * B / F).
# ---------------------------------------------------------------------------


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def phys_block(k, di, S):
    return k if di == 0 else (S - 2 - k)


def dwell_station(k, di, S):
    return k if di == 0 else (S - 1 - k)


def serialized_baseline(S, trains):
    # fully serialized dispatch in release order: each train departs only once
    # the previous has cleared the whole line.  Always feasible (no meets, no
    # dwell); this is the checker's trivial reference construction.
    order = sorted(range(len(trains)), key=lambda j: (trains[j][1], j))
    clear = 0
    F = 0
    for j in order:
        di, r, d, w, v, h = trains[j]
        dep = max(r, clear)
        arr = dep + (S - 1) * h
        clear = arr
        F += w * (arr - r) + v * max(0, arr - d)
    return F


def main():
    toks = open(sys.argv[1]).read().split()
    it = iter(toks)
    S = int(next(it)); TMAX = int(next(it))
    cap = [int(next(it)) for _ in range(S)]
    N = int(next(it))
    trains = []
    for _ in range(N):
        di = int(next(it)); r = int(next(it)); d = int(next(it))
        w = int(next(it)); v = int(next(it)); h = int(next(it))
        trains.append((di, r, d, w, v, h))

    raw = open(sys.argv[2]).read().split()
    need = N * (S - 1)
    if len(raw) != need:
        fail("token count %d != %d" % (len(raw), need))
    vals = []
    for tk in raw:
        try:
            x = int(tk)
        except Exception:
            fail("non-integer token %r" % tk)
        if not math.isfinite(x):
            fail("non-finite")
        vals.append(x)

    block_iv = [[] for _ in range(S - 1)]
    stn_iv = [[] for _ in range(S)]
    F = 0
    pos = 0
    for j, (di, r, d, w, v, h) in enumerate(trains):
        e = vals[pos:pos + (S - 1)]
        pos += (S - 1)
        for k in range(S - 1):
            if e[k] < 0 or e[k] > TMAX:
                fail("entry out of range")
        if e[0] < r:
            fail("departs before release (train %d)" % j)
        for k in range(S - 2):
            if e[k + 1] < e[k] + h:
                fail("block %d->%d too fast (train %d)" % (k, k + 1, j))
        for k in range(S - 1):
            b = phys_block(k, di, S)
            block_iv[b].append((e[k], e[k] + h))
            if k >= 1:
                st = dwell_station(k, di, S)
                ds = e[k - 1] + h
                de = e[k]
                if de > ds:
                    stn_iv[st].append((ds, de))
        arr = e[S - 2] + h
        F += w * (arr - r) + v * max(0, arr - d)

    for b in range(S - 1):
        iv = sorted(block_iv[b])
        for t in range(1, len(iv)):
            if iv[t][0] < iv[t - 1][1]:
                fail("block %d double-occupied" % b)

    for st in range(S):
        iv = stn_iv[st]
        if not iv:
            continue
        ev = []
        for (s, e2) in iv:
            ev.append((s, 1)); ev.append((e2, -1))
        ev.sort()
        cur = 0; mx = 0
        for (_, dl) in ev:
            cur += dl
            if cur > mx:
                mx = cur
        if mx > cap[st]:
            fail("siding %d overflow (%d > %d)" % (st, mx, cap[st]))

    B = serialized_baseline(S, trains)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
