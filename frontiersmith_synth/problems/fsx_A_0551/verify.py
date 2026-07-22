#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  ->  prints 'Ratio: <float in [0,1]>'.

Scores an erosion-CA sediment-delta problem (format C, maximize).

Objective F = sediment mass the CA drops inside the target zone after the
solver's edits.  Baseline B = the deposition of a canonical routing
construction the checker builds itself (route the spring to the zone with a
mild stall).  Maximization normalization:  sc = min(1000, 100*F/max(1e-9,B)),
Ratio = sc/1000  (trivial reproduces B -> 0.1, 10x-better caps at 1.0).

The physics rewards the DERIVATIVE of the slope: capacity = CAP_K*slope, so
sediment is entrained where the slope steepens and dropped where it flattens.
"""
import sys


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def parse_instance(path):
    tok = read_ints(path)
    it = iter(tok)
    def nxt():
        return int(next(it))
    N = nxt(); V = nxt(); T = nxt(); CAP_K = nxt(); EROSION = nxt()
    DEPOSIT = nxt(); HMAX = nxt(); SEA = nxt()
    sy = nxt(); sx = nxt()
    zy0 = nxt(); zx0 = nxt(); zy1 = nxt(); zx1 = nxt()
    h = [[nxt() for _ in range(N)] for _ in range(N)]
    params = dict(N=N, V=V, T=T, CAP_K=CAP_K, EROSION=EROSION, DEPOSIT=DEPOSIT,
                  HMAX=HMAX, SEA=SEA, sy=sy, sx=sx,
                  zy0=zy0, zx0=zx0, zy1=zy1, zx1=zx1)
    return params, h


def zone_set(p):
    return {(y, x) for y in range(p["zy0"], p["zy1"] + 1)
                    for x in range(p["zx0"], p["zx1"] + 1)}


NB = ((-1, 0), (0, 1), (1, 0), (0, -1))     # fixed tie-break order


def one_pass(g, p, zset):
    """One water pulse of the integer transport CA over the FIXED field g
    (terrain is not rewritten -- a passive sediment tracer).  Water descends by
    steepest descent from the spring; capacity = CAP_K*slope; it entrains where
    capacity exceeds its load and deposits where capacity is below it.  Returns
    sediment dropped inside the zone this pass."""
    N = p["N"]; CAP_K = p["CAP_K"]
    EROSION = p["EROSION"]; DEPOSIT = p["DEPOSIT"]; SEA = p["SEA"]
    sy = p["sy"]; sx = p["sx"]
    MAXSTEPS = N * N + 8
    L = 0
    dep = 0
    y, x = sy, sx
    for _s in range(MAXSTEPS):
        if g[y][x] <= SEA:                    # reached the sea -> water + load exit
            break
        best = None; bh = None
        for dy, dx in NB:
            ny, nx = y + dy, x + dx
            if 0 <= ny < N and 0 <= nx < N:
                hv = g[ny][nx]
                if bh is None or hv < bh:
                    bh = hv; best = (ny, nx)
        if best is None or bh >= g[y][x]:     # pit / no downhill -> pool, drop all
            if (y, x) in zset:
                dep += L
            break
        slope = g[y][x] - bh
        cap = CAP_K * slope
        if cap > L:                           # spare capacity -> entrain
            e = cap - L
            if e > EROSION:
                e = EROSION
            room = g[y][x] - (SEA + 1)
            if e > room:
                e = room
            if e > 0:
                L += e
        elif cap < L:                         # over capacity -> deposit
            d = L - cap
            if d > DEPOSIT:
                d = DEPOSIT
            if (y, x) in zset:
                dep += d
            L -= d
        y, x = best
    return dep


def simulate(h, p, zset):
    """Total zone deposition over T identical pulses on the fixed field.  O(T*N)."""
    return p["T"] * one_pass(h, p, zset)


def canonical_path(p):
    """Spring -> zone column (horizontal feeder) -> straight down to the coast.
    Shared by the checker baseline and the trivial reference."""
    N = p["N"]; sy = p["sy"]; sx = p["sx"]
    zcx = (p["zx0"] + p["zx1"]) // 2
    path = []
    x = sx
    while x != zcx:
        path.append((sy, x)); x += 1 if zcx > sx else -1
    for y in range(sy, N):
        path.append((y, zcx))
    return path, zcx


def carve(g, path, prof, N, HMAX):
    """Set channel heights to prof (made non-increasing) and raise the immediate
    off-channel neighbours into walls, so the water is forced to follow the
    channel to the sea (deterministic routing)."""
    pf = list(prof)
    for i in range(1, len(pf)):
        if pf[i] >= pf[i - 1]:
            pf[i] = pf[i - 1] - 1
        if pf[i] < 0:
            pf[i] = 0
    pset = set(path)
    for i, (y, x) in enumerate(path):
        g[y][x] = pf[i]
    for (y, x) in path:
        for dy, dx in NB:
            ny, nx = y + dy, x + dx
            if 0 <= ny < N and 0 <= nx < N and (ny, nx) not in pset:
                if g[ny][nx] <= g[y][x]:
                    g[ny][nx] = min(HMAX, g[y][x] + 1)


def baseline_field(p, h):
    """Canonical routing construction the checker scores as B: route the spring
    to the zone and give the zone a MILD stall (gentle slope 1).  No slope
    engineering.  Deterministic and positive on every test."""
    N = p["N"]; sy = p["sy"]; HMAX = p["HMAX"]
    zy0 = p["zy0"]; zy1 = p["zy1"]
    g = [row[:] for row in h]
    path, zcx = canonical_path(p)
    start = h[sy][p["sx"]]
    lin = [(start * (N - 1 - y)) // (N - 1) for y in range(N)]
    tgt = {}
    for (y, x) in path:
        tgt[(y, x)] = lin[y] if y >= sy else start
    # steep entry so the flattening (and its deposition) lands INSIDE the zone,
    # then a mild slope-1 stall across the zone rows
    ztop = max(1, (lin[zy0 - 1] if zy0 - 1 >= 0 else start) - 7)
    for k, y in enumerate(range(zy0, zy1 + 1)):        # mild stall, slope 1
        tgt[(y, zcx)] = max(0, ztop - k)
    bot = tgt[(zy1, zcx)]
    for k, y in enumerate(range(zy1 + 1, N)):
        span = max(1, N - 1 - zy1)
        tgt[(y, zcx)] = max(0, bot - (bot * (k + 1)) // span)
    prof = [tgt[c] for c in path]
    carve(g, path, prof, N, HMAX)
    return g


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    p, h0 = parse_instance(inf)
    N = p["N"]; V = p["V"]; HMAX = p["HMAX"]

    # ---- parse participant edits strictly ----
    try:
        with open(outf) as f:
            toks = f.read().split()
    except Exception:
        fail("unreadable output")
    if not toks:
        fail("empty output")

    def as_int(t):
        # reject nan/inf/floats/garbage -> only plain integers allowed
        if t and (t[0] in "+-"):
            body = t[1:]
        else:
            body = t
        if not body.isdigit():
            raise ValueError("non-int token %r" % t)
        return int(t)

    try:
        K = as_int(toks[0])
    except ValueError as e:
        fail(str(e))
    if K < 0 or K > N * N:
        fail("bad edit count %d" % K)
    if len(toks) < 1 + 3 * K:
        fail("truncated edit list")

    g = [row[:] for row in h0]
    budget = 0
    idx = 1
    for _ in range(K):
        try:
            y = as_int(toks[idx]); x = as_int(toks[idx + 1]); d = as_int(toks[idx + 2])
        except ValueError as e:
            fail(str(e))
        idx += 3
        if not (0 <= y < N and 0 <= x < N):
            fail("edit out of grid (%d,%d)" % (y, x))
        budget += abs(d)
        if budget > V:
            fail("edit budget exceeded (%d > %d)" % (budget, V))
        g[y][x] += d
        if g[y][x] < 0 or g[y][x] > HMAX:
            fail("cell (%d,%d) height %d out of [0,%d]" % (y, x, g[y][x], HMAX))

    zset = zone_set(p)
    F = simulate(g, p, zset)
    B = simulate(baseline_field(p, h0), p, zset)
    B = max(1e-9, float(B))

    sc = min(1000.0, 100.0 * float(F) / B)
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
