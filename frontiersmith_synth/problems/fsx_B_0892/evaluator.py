import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0892 -- metamaterial-field-layout (Format B, isolated candidate)
# Theme: "Lay out unit cells for a target property field" -- a graded-index
# metamaterial panel is tiled from a fixed library of unit-cell types. Each
# type has a known inclusion volume fraction; its EFFECTIVE permittivity is
# given by the Maxwell-Garnett homogenization mixing rule. The candidate must
# choose one tile type per grid cell so the achieved effective-property field
# tracks a target field, while adjacent tiles stay index-compatible (a large
# jump in inclusion fraction between neighboring cells causes interface
# delamination / scattering loss in the physical panel).
# Objective: MINIMIZE mean pointwise field error + mean adjacency (interface)
# penalty. This is a joint labeling problem with a smoothness term -- pick
# each cell's tile independently by nearest match and you win on field error
# but pay heavily in interface cost at any region where the target is noisy
# or has a sharp local feature; the strong move is to solve it like an
# MRF/graph-cut, trading a little pointwise accuracy for interface economy.
# ==========================================================================


def _mg_eff(v, em, ei):
    """Maxwell-Garnett effective permittivity of a v-fraction inclusion
    (permittivity ei) embedded in a matrix (permittivity em)."""
    return em * (2 * (1 - v) * em + (1 + 2 * v) * ei) / ((2 + v) * em + (1 - v) * ei)


def make_instances():
    K = 7
    C = 9
    out = []
    # idx 1,2,3,4,6,7,8 are "trap" instances: a planted sharp checkerboard
    # feature plus mild target noise, and a heavier interface weight -- the
    # obvious per-cell nearest-match falls apart here. idx 0,5,9 are calmer
    # held-out instances (smooth field, light interface weight) where nearest
    # match is already close to fine, so the ladder isn't trap-only.
    trap_idxs = {1, 2, 3, 4, 6, 7, 8}
    for idx in range(10):
        seed = 5000 + idx
        rng = random.Random(seed)
        R = 6 + (idx % 2)
        v_table = sorted(round(0.05 + 0.9 * i / (K - 1) + rng.uniform(-0.01, 0.01), 4)
                          for i in range(K))
        em = round(1.0 + 1.0 * rng.random(), 3)
        ei = round(6.0 + 5.0 * rng.random(), 3)
        eff_table = [round(_mg_eff(v, em, ei), 6) for v in v_table]
        lo, hi = eff_table[0], eff_table[-1]

        trap = idx in trap_idxs
        cr = R / 2 + rng.uniform(-1, 1)
        cc = C / 2 + rng.uniform(-1, 1)
        maxrho = math.hypot(max(cr, R - cr), max(cc, C - cc))

        T = [[0.0] * C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                rho = math.hypot(r - cr, c - cc) / (maxrho + 1e-9)
                T[r][c] = hi - (hi - lo) * rho * rho   # graded-index lens profile

        if trap:
            pr0 = rng.randrange(0, max(1, R - 3))
            pc0 = rng.randrange(0, max(1, C - 3))
            for dr in range(3):
                for dc in range(3):
                    r, c = pr0 + dr, pc0 + dc
                    if r < R and c < C:
                        T[r][c] = hi if (dr + dc) % 2 == 0 else lo

        noise_sigma = (hi - lo) * (0.05 if trap else 0.02)
        for r in range(R):
            for c in range(C):
                T[r][c] = round(T[r][c] + rng.gauss(0, noise_sigma), 6)

        lam = 0.5 if trap else 0.12
        pub = {
            "R": R, "C": C, "K": K,
            "v_table": v_table, "em": em, "ei": ei, "eff_table": eff_table,
            "target": T, "interface_weight": lam,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _objective(pub, types):
    R, C, K = pub["R"], pub["C"], pub["K"]
    eff_table = pub["eff_table"]
    T = pub["target"]
    lam = pub["interface_weight"]
    u = 0.0
    for r in range(R):
        for c in range(C):
            u += abs(eff_table[types[r][c]] - T[r][c])
    u /= (R * C)
    e = 0.0
    ne = 0
    for r in range(R):
        for c in range(C):
            if c + 1 < C:
                d = types[r][c] - types[r][c + 1]; e += d * d; ne += 1
            if r + 1 < R:
                d = types[r][c] - types[r + 1][c]; e += d * d; ne += 1
    e /= ne
    return u + lam * e


def baseline(inst):
    """Best single tile type applied uniformly across the whole grid (the
    'lay one tile everywhere' reference construction) -- zero interface
    cost, pure pointwise-optimal constant."""
    pub = inst["public"]
    R, C, K = pub["R"], pub["C"], pub["K"]
    eff_table = pub["eff_table"]
    T = pub["target"]
    flat = [T[r][c] for r in range(R) for c in range(C)]
    best_t = min(range(K), key=lambda t: sum(abs(eff_table[t] - x) for x in flat))
    types = [[best_t] * C for _ in range(R)]
    return _objective(pub, types)


def score(inst, ans):
    pub = inst["public"]
    R, C, K = pub["R"], pub["C"], pub["K"]
    if not isinstance(ans, dict) or "types" not in ans:
        return False, 0.0
    grid = ans["types"]
    if not isinstance(grid, list) or len(grid) != R:
        return False, 0.0
    types = []
    for row in grid:
        if not isinstance(row, list) or len(row) != C:
            return False, 0.0
        clean_row = []
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, 0.0
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
            iv = int(round(v))
            if abs(v - iv) > 1e-6 or iv < 0 or iv >= K:
                return False, 0.0
            clean_row.append(iv)
        types.append(clean_row)
    obj = _objective(pub, types)
    if obj != obj or obj <= 0.0:
        return False, 0.0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=8)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
