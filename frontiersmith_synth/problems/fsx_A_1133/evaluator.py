import sys, json, random, isorun
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Symmetry-group machinery (frozen, evaluator-side ground truth).
# ---------------------------------------------------------------------------
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


def perm_of(N, kind, p=None):
    arr = [0] * (N * N)
    for r in range(N):
        for c in range(N):
            if kind == "r90":
                nr, nc = c, N - 1 - r
            elif kind == "r180":
                nr, nc = N - 1 - r, N - 1 - c
            elif kind == "r270":
                nr, nc = N - 1 - c, r
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


def build_group(N, pg_name, p):
    gens = [perm_of(N, k) for k in POINT_GROUPS[pg_name]]
    if p != N:
        gens.append(perm_of(N, "tx", p))
        gens.append(perm_of(N, "ty", p))
    return closure(gens, N * N)


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
    return orbit_id, members


# ---------------------------------------------------------------------------
# Instance generation (deterministic, seeded).
# ---------------------------------------------------------------------------
# Each combo is chosen so the fundamental-domain orbit count (~ p^2/|H| for a
# translating group, ~ N^2/|H| for a pure point group) stays in a rich-but-solvable
# range -- large point groups (order >=4) are paired with p=N (no translation, so the
# motif isn't folded down to a handful of giant orbits); small point groups (order
# <=2) may carry a genuine periodic-translation lattice on top without collapsing.
INSTANCE_SPECS = [
    dict(N=8,  K=4, pg="D4",     p=8,  pattern="uniform",     d=0.30, seed=1001),
    dict(N=10, K=4, pg="C2_180", p=5,  pattern="orbit-block", d=0.35, seed=1002),
    dict(N=9,  K=3, pg="D2ax",   p=9,  pattern="uniform",     d=0.25, seed=1003),
    dict(N=12, K=4, pg="C4",     p=12, pattern="orbit-block", d=0.40, seed=1004),
    dict(N=10, K=4, pg="D2di",   p=10, pattern="uniform",     d=0.30, seed=1005),
    dict(N=10, K=5, pg="C1",     p=5,  pattern="orbit-block", d=0.45, seed=1006),
    dict(N=12, K=3, pg="C2_fd",  p=12, pattern="uniform",     d=0.28, seed=1007),
    dict(N=12, K=4, pg="D4",     p=12, pattern="uniform",     d=0.35, seed=1008),
    dict(N=12, K=4, pg="C2_fh",  p=6,  pattern="orbit-block", d=0.38, seed=1009),
    dict(N=12, K=5, pg="C2_180", p=6,  pattern="orbit-block", d=0.42, seed=1010),
]


def build_instance(spec):
    N, K, pg, p = spec["N"], spec["K"], spec["pg"], spec["p"]
    pattern, d, seed = spec["pattern"], spec["d"], spec["seed"]
    n = N * N
    group = build_group(N, pg, p)
    orbit_id, members = orbits_of(group, n)
    rng = random.Random(seed)
    orbit_color = [rng.randrange(K) for _ in members]
    grid_true = [0] * n
    for oid, mem in enumerate(members):
        cval = orbit_color[oid]
        for idx in mem:
            grid_true[idx] = cval

    damaged = [False] * n
    rng2 = random.Random(seed * 7 + 3)
    if pattern == "uniform":
        for idx in range(n):
            if rng2.random() < d:
                damaged[idx] = True
        # Even scattered iid damage should leave at least a couple of orbits with
        # zero surviving evidence (so the local-statistics fallback is genuinely
        # exercised); large orbits almost never get fully wiped by iid noise alone,
        # so top up deterministically if none happened to be fully erased.
        multi = [mem for mem in members if len(mem) >= 2]
        fully_erased = sum(1 for mem in multi if all(damaged[i] for i in mem))
        if multi and fully_erased == 0:
            pick_order = list(range(len(multi)))
            rng2.shuffle(pick_order)
            for k in pick_order[:min(2, len(multi))]:
                for idx in multi[k]:
                    damaged[idx] = True
    else:  # orbit-block: erase whole orbits first (guarantees fully-erased orbits)
        order = list(range(len(members)))
        rng2.shuffle(order)
        target = d * n
        erased_cnt = 0
        for oid in order:
            if erased_cnt >= target:
                break
            mem = members[oid]
            for idx in mem:
                damaged[idx] = True
            erased_cnt += len(mem)
        for idx in range(n):
            if not damaged[idx] and rng2.random() < 0.05:
                damaged[idx] = True

    grid_pub = []
    for r in range(N):
        row = [(-1 if damaged[r * N + c] else grid_true[r * N + c]) for c in range(N)]
        grid_pub.append(row)

    public = {"name": "ornament_%d" % seed, "n": N, "k": K, "grid": grid_pub}
    hidden = {"n": N, "k": K,
              "grid_true": [grid_true[r * N:(r + 1) * N] for r in range(N)],
              "orbit_id": orbit_id}
    return {"public": public, "hidden": hidden}


def make_instances():
    return [build_instance(s) for s in INSTANCE_SPECS]


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def quality(inst, cand_flat):
    hidden = inst["hidden"]
    N, K = hidden["n"], hidden["k"]
    true_flat = [v for row in hidden["grid_true"] for v in row]
    orbit_id = hidden["orbit_id"]
    pub_flat = [v for row in inst["public"]["grid"] for v in row]

    orb_members = defaultdict(list)
    for idx, oid in enumerate(orbit_id):
        orb_members[oid].append(idx)

    det_correct = 0
    det_total = 0
    erased_multi_orbits = []   # orbits (size>=2) with zero known cells
    erased_cells = []          # all cells belonging to a fully-erased orbit

    for mem in orb_members.values():
        known = [idx for idx in mem if pub_flat[idx] != -1]
        if known:
            for idx in mem:
                det_total += 1
                if cand_flat[idx] == true_flat[idx]:
                    det_correct += 1
        else:
            erased_cells.extend(mem)
            if len(mem) >= 2:
                erased_multi_orbits.append(mem)

    accuracy_det = (det_correct / det_total) if det_total > 0 else 1.0

    if erased_multi_orbits:
        good = sum(1 for mem in erased_multi_orbits
                   if len(set(cand_flat[idx] for idx in mem)) == 1)
        consistency_bonus = good / len(erased_multi_orbits)
    else:
        consistency_bonus = 1.0

    if erased_cells:
        tot = len(erased_cells)
        th = [0] * K
        ch = [0] * K
        for idx in erased_cells:
            th[true_flat[idx]] += 1
            cv = cand_flat[idx]
            if 0 <= cv < K:
                ch[cv] += 1
        tv = 0.5 * sum(abs(th[k] / tot - ch[k] / tot) for k in range(K))
        stat_preservation = 1.0 - tv
    else:
        stat_preservation = 1.0

    return 0.55 * accuracy_det + 0.20 * consistency_bonus + 0.25 * stat_preservation


def score(inst, ans):
    if not isinstance(ans, dict) or "grid" not in ans:
        return False, 0.0
    grid = ans["grid"]
    N, K = inst["public"]["n"], inst["public"]["k"]
    if not isinstance(grid, list) or len(grid) != N:
        return False, 0.0
    flat = []
    for row in grid:
        if not isinstance(row, list) or len(row) != N:
            return False, 0.0
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return False, 0.0
            if not (0 <= v < K):
                return False, 0.0
            flat.append(v)
    pub_flat = [v for row in inst["public"]["grid"] for v in row]
    for idx, pv in enumerate(pub_flat):
        if pv != -1 and flat[idx] != pv:
            return False, 0.0
    q = quality(inst, flat)
    if not (q == q):
        return False, 0.0
    return True, q


def baseline(inst):
    N, K = inst["public"]["n"], inst["public"]["k"]
    pub_flat = [v for row in inst["public"]["grid"] for v in row]
    known = [v for v in pub_flat if v != -1]
    if known:
        cnt = Counter(known)
        mode = max(range(K), key=lambda k: cnt.get(k, 0))
    else:
        mode = 0
    flat = [v if v != -1 else mode for v in pub_flat]
    return quality(inst, flat)


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, q = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        qb = baseline(inst)
        denom = max(1e-9, 1.0 - qb)
        r = 0.1 + 0.9 * (q - qb) / denom
        if r != r:
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
