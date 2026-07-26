# TIER: strong
# Group inference under ambiguity (Occam in the subgroup lattice): search the full
# candidate lattice of point-symmetry x periodic-translation groups, keep the LARGEST
# one fully consistent with every visible pair, then orbit-close known cells across it.
# Orbits with zero surviving evidence are fundamentally unidentifiable per-cell -- fill
# them from the ornament's own global color statistics (proportional, not a fixed
# default), preserving the motif's color mix instead of guessing arbitrarily.
import sys, json
from collections import Counter

POINT_GROUPS = {
    "D4":     ["r90", "fh"],
    "C4":     ["r90"],
    "D2ax":   ["r180", "fh"],
    "D2di":   ["r180", "fd"],
    "C2_180": ["r180"],
    "C2_fh":  ["fh"],
    "C2_fv":  ["fv"],
    "C2_fd":  ["fd"],
    "C2_fa":  ["fa"],
    "C1":     [],
}
ORDER_PRIORITY = ["D4", "C4", "D2ax", "D2di", "C2_180",
                   "C2_fh", "C2_fv", "C2_fd", "C2_fa", "C1"]


def perm_of(N, kind, p=None):
    arr = [0] * (N * N)
    for r in range(N):
        for c in range(N):
            if kind == "r90":
                nr, nc = c, N - 1 - r
            elif kind == "r180":
                nr, nc = N - 1 - r, N - 1 - c
            elif kind == "fh":
                nr, nc = N - 1 - r, c
            elif kind == "fv":
                nr, nc = r, N - 1 - c
            elif kind == "fd":
                nr, nc = c, r
            elif kind == "fa":
                nr, nc = N - 1 - c, N - 1 - r
            elif kind == "tx":
                nr, nc = (r + p) % N, c
            elif kind == "ty":
                nr, nc = r, (c + p) % N
            else:
                nr, nc = r, c
            arr[r * N + c] = nr * N + nc
    return tuple(arr)


def closure(gens, n):
    ident = tuple(range(n))
    seen = {ident}
    frontier = [ident]
    while frontier:
        nf = []
        for q in frontier:
            for g in gens:
                nq = tuple(g[q[i]] for i in range(n))
                if nq not in seen:
                    seen.add(nq)
                    nf.append(nq)
        frontier = nf
    return seen


def orbits_of(group, n):
    glist = list(group)
    orbit_id = [-1] * n
    members = []
    nid = 0
    for i in range(n):
        if orbit_id[i] != -1:
            continue
        mem = sorted(set(g[i] for g in glist))
        for m in mem:
            orbit_id[m] = nid
        members.append(mem)
        nid += 1
    return members


def divisors(N):
    return [d for d in range(1, N + 1) if N % d == 0]


def candidate_periods(N):
    ps = [p for p in divisors(N) if p >= 2 and (N // p) <= 3]
    return ps if ps else [N]


def main():
    inst = json.load(sys.stdin)
    N, K, grid = inst["n"], inst["k"], inst["grid"]
    flat = [grid[r][c] for r in range(N) for c in range(N)]
    n = N * N
    periods = candidate_periods(N)

    best_order = -1
    best_members = None
    for pg_name in ORDER_PRIORITY:
        gens = [perm_of(N, k) for k in POINT_GROUPS[pg_name]]
        for p in periods:
            g = list(gens)
            if p != N:
                g.append(perm_of(N, "tx", p))
                g.append(perm_of(N, "ty", p))
            group = closure(g, n)
            order = len(group)
            if order <= best_order:
                continue
            members = orbits_of(group, n)
            consistent = True
            for mem in members:
                vals = set(flat[i] for i in mem if flat[i] != -1)
                if len(vals) > 1:
                    consistent = False
                    break
            if consistent:
                best_order = order
                best_members = members

    if best_members is None:
        best_members = [[i] for i in range(n)]  # identity fallback (always consistent)

    known_global = [v for v in flat if v != -1]
    hist = Counter(known_global)
    total_known = sum(hist.values())

    out = list(flat)
    erased_orbits = []
    for mem in best_members:
        known_vals = [flat[i] for i in mem if flat[i] != -1]
        if known_vals:
            v = known_vals[0]
            for i in mem:
                out[i] = v
        else:
            erased_orbits.append(mem)

    if erased_orbits:
        if total_known > 0:
            props = [hist.get(k, 0) / total_known for k in range(K)]
        else:
            props = [1.0 / K] * K
        m = len(erased_orbits)
        raw = [props[k] * m for k in range(K)]
        base_counts = [int(x) for x in raw]
        remainders = sorted(range(K), key=lambda k: raw[k] - base_counts[k], reverse=True)
        rem = m - sum(base_counts)
        i = 0
        while rem > 0 and K > 0:
            base_counts[remainders[i % K]] += 1
            rem -= 1
            i += 1
        colors = []
        for k in range(K):
            colors.extend([k] * base_counts[k])
        while len(colors) < m:
            colors.append(0)
        colors = colors[:m]
        order_idx = sorted(range(m), key=lambda oi: (-len(erased_orbits[oi]), min(erased_orbits[oi])))
        for rank, oi in enumerate(order_idx):
            v = colors[rank]
            for i in erased_orbits[oi]:
                out[i] = v

    grid_out = [[out[r * N + c] for c in range(N)] for r in range(N)]
    print(json.dumps({"grid": grid_out}))


main()
