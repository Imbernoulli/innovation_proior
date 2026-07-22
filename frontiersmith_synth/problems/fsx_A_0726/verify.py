#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -> prints 'Ratio: <x in [0,1]>' (last line authoritative).

Carved-percolation-throughput: score = total steady-state flow from the row-0 aquifer to
the single outlet cistern, under a FIXED-ITERATION integer pressure relaxation over a
resistor-style conductance network built from the participant's carved cell set.

Conductance rules (all integer, deterministic):
  - exactly one endpoint carved (an interface edge, tunnel wall against rock):
    conductance = the ROCK side's base permeability -- this is where nearly all the
    resistance in the system concentrates (a carved cell only ever drinks from rock at
    ordinary rock permeability; there is no free unlimited tap anywhere).
  - neither carved: conductance = min(perm) of the two rock cells (slow diffusion
    through plain rock).
  - both endpoints carved: high (C_TUNNEL) for ordinary tunnel continuation, UNLESS the
    edge is a "merge" edge: let dist(x) = the shortest hop-count from x to the outlet
    cell through carved cells only (BFS; the outlet is the root). An edge (u,v) with
    dist(u) = dist(v)+1 feeds v from upstream. If v has >=2 such upstream carved
    neighbours (several tunnel branches converging on v), EACH of those upstream edges
    is throttled: cond = C_TUNNEL // upcount(v). A trunk's own downstream continuation
    is a *different* edge (evaluated at the next node down) and is untouched -- only the
    actual convergence point pays the turbulent merge loss, not the whole corridor.
Row 0 (the aquifer) and the single outlet cell are ALWAYS fixed-pressure boundary nodes;
their own base permeability is ordinary rock too (no special-casing), so "carving" them
is legal but never beneficial.

Any feasibility violation (bad token count, out of range, duplicate, non-integer,
non-finite, budget overrun) prints 'Ratio: 0.0' and exits 0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    i = 0
    R, C, B, r_out, c_out = (int(toks[i + k]) for k in range(5)); i += 5
    P_IN, P_OUT, C_TUNNEL, ITERS = (int(toks[i + k]) for k in range(4)); i += 4
    perm = []
    for r in range(R):
        row = [int(toks[i + k]) for k in range(C)]
        i += C
        perm.append(row)
    return R, C, B, r_out, c_out, P_IN, P_OUT, C_TUNNEL, ITERS, perm


def neighbors4(r, c, R, C):
    if r > 0:
        yield r - 1, c
    if r < R - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < C - 1:
        yield r, c + 1


def compute_flow(R, C, r_out, c_out, P_IN, P_OUT, C_TUNNEL, ITERS, perm, carved):
    """Fixed-iteration integer Jacobi relaxation; returns total flow into the outlet."""
    # BFS hop-distance from the outlet, through carved cells only (outlet = root, dist 0)
    from collections import deque
    dist = {(r_out, c_out): 0}
    dq = deque([(r_out, c_out)])
    while dq:
        cell = dq.popleft()
        r, c = cell
        for nr, nc in neighbors4(r, c, R, C):
            if (nr, nc) in carved and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[cell] + 1
                dq.append((nr, nc))

    upcount = {}
    for (r, c) in carved:
        d = dist.get((r, c))
        if d is None:
            continue
        cnt = 0
        for nr, nc in neighbors4(r, c, R, C):
            if (nr, nc) in carved and dist.get((nr, nc)) == d + 1:
                cnt += 1
        upcount[(r, c)] = cnt

    def cond(u, v):
        cu = u in carved
        cv = v in carved
        if cu and cv:
            du, dv = dist.get(u), dist.get(v)
            if du is not None and dv is not None and abs(du - dv) == 1:
                vdown = u if du < dv else v
                uc = upcount.get(vdown, 1)
                if uc >= 2:
                    return C_TUNNEL // uc
            return C_TUNNEL
        pu = C_TUNNEL if cu else perm[u[0]][u[1]]
        pv = C_TUNNEL if cv else perm[v[0]][v[1]]
        return pu if pu < pv else pv

    # precompute neighbor lists + conductances once (grid + carved set are both fixed)
    nbcond = [[None] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            lst = []
            for nr, nc in neighbors4(r, c, R, C):
                lst.append((nr, nc, cond((r, c), (nr, nc))))
            nbcond[r][c] = lst

    P = [[0] * C for _ in range(R)]
    for c in range(C):
        P[0][c] = P_IN
    P[r_out][c_out] = P_OUT
    fixed = [[False] * C for _ in range(R)]
    for c in range(C):
        fixed[0][c] = True
    fixed[r_out][c_out] = True

    for _ in range(ITERS):
        newP = [row[:] for row in P]
        for r in range(1, R):
            frow = fixed[r]
            prow = newP[r]
            for c in range(C):
                if frow[c]:
                    continue
                sc = 0
                sf = 0
                for nr, nc, cd in nbcond[r][c]:
                    sc += cd
                    sf += cd * P[nr][nc]
                if sc > 0:
                    prow[c] = sf // sc
        P = newP

    flow = 0
    for nr, nc, cd in nbcond[r_out][c_out]:
        flow += cd * (P[nr][nc] - P[r_out][c_out])
    return flow if flow > 0 else 0


def baseline_carved(R, C, B, r_out, c_out):
    """Checker's own trivial construction: a straight 1-wide line directly above the
    outlet, using min(B, R-1) cells; leftover budget is simply not used."""
    s = set()
    r = r_out - 1
    used = 0
    while r >= 1 and used < B and used < R - 1:
        s.add((r, c_out))
        r -= 1
        used += 1
    return s


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inp, outp = sys.argv[1], sys.argv[2]
    R, C, B, r_out, c_out, P_IN, P_OUT, C_TUNNEL, ITERS, perm = read_instance(inp)

    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if not raw:
        fail("empty output")
    try:
        k = int(raw[0])
    except ValueError:
        fail("first token not an integer count")
    if k < 0 or k > B:
        fail("carve count %d out of [0,%d]" % (k, B))
    if len(raw) != 1 + 2 * k:
        fail("expected %d tokens, got %d" % (1 + 2 * k, len(raw)))

    carved = set()
    for j in range(k):
        try:
            r = int(raw[1 + 2 * j])
            c = int(raw[2 + 2 * j])
        except ValueError:
            fail("non-integer coordinate")
        if r < 0 or r >= R or c < 0 or c >= C:
            fail("coordinate (%d,%d) out of grid" % (r, c))
        if (r, c) in carved:
            fail("duplicate coordinate (%d,%d)" % (r, c))
        carved.add((r, c))

    F = compute_flow(R, C, r_out, c_out, P_IN, P_OUT, C_TUNNEL, ITERS, perm, carved)

    base_set = baseline_carved(R, C, B, r_out, c_out)
    Bflow = compute_flow(R, C, r_out, c_out, P_IN, P_OUT, C_TUNNEL, ITERS, perm, base_set)

    sc = min(1000.0, 100.0 * F / max(1e-9, float(Bflow)))
    ratio = sc / 1000.0
    sys.stdout.write("flow F=%d baseline B=%d carved=%d\nRatio: %.6f\n" % (F, Bflow, k, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
