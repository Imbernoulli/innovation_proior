import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0865 -- block-reordering-balancer (Format B, isolated candidate)
# Theme: iterative matrix balancing (Sinkhorn/RAS) accelerated on nearly
# block-decomposable instances within a fixed OPERATION budget (not
# wall-clock -- the evaluator deterministically REPLAYS the candidate's
# submitted move list, so cost = len(ops), independent of machine speed).
#
# Public instance: an n x n strictly-positive coupling matrix M (rows/cols
# possibly hidden-block structured after a secret permutation) plus target
# row sums r and column sums c, and an operation budget B.
#
# Answer: a list of at most B "moves". Each move is one of:
#   {"type":"row",  "omega": w}                         -- full row sweep
#   {"type":"col",  "omega": w}                         -- full col sweep
#   {"type":"block","axis":"row"|"col","omega":w,"groups":[[...],...]}
#       -- coarse block-level correction: partition {0..n-1} into groups,
#          and rescale EVERY row (or column) in a group by ONE shared
#          multiplier that corrects only the group's AGGREGATE imbalance.
# All moves are SOR-style (omega=1 is the plain exact projection; omega>1
# over-relaxes / extrapolates the step). The evaluator replays the moves
# from the identity scaling and scores the final residual imbalance
# (mean relative L1 deviation of achieved row/col sums from targets).
# Objective: MINIMIZE. An invalid/malformed answer scores 0.
# ==========================================================================

OMEGA_LO, OMEGA_HI = 0.0, 3.0
CLIP_LO, CLIP_HI = 1e-9, 1e9


def _dot_row(M, i, v, n):
    row = M[i]
    return sum(row[j] * v[j] for j in range(n))


def _dot_col(M, j, u, n):
    return sum(M[i][j] * u[i] for i in range(n))


def replay(pub, ops):
    """Deterministically replay a validated move list from identity scaling.
    Returns (imbalance, u, v) or None on numerical blow-up."""
    n = pub["n"]
    M = pub["M"]; r = pub["r"]; c = pub["c"]
    u = [1.0] * n
    v = [1.0] * n
    for op in ops:
        t = op["type"]
        w = op["omega"]
        if t == "row":
            for i in range(n):
                s = u[i] * _dot_row(M, i, v, n)
                if s <= 0 or s != s:
                    return None
                m = (r[i] / s) ** w
                m = min(max(m, CLIP_LO), CLIP_HI)
                u[i] = min(max(u[i] * m, CLIP_LO), CLIP_HI)
        elif t == "col":
            for j in range(n):
                s = v[j] * _dot_col(M, j, u, n)
                if s <= 0 or s != s:
                    return None
                m = (c[j] / s) ** w
                m = min(max(m, CLIP_LO), CLIP_HI)
                v[j] = min(max(v[j] * m, CLIP_LO), CLIP_HI)
        elif t == "block":
            groups = op["groups"]
            if op["axis"] == "row":
                for g in groups:
                    S = sum(u[i] * _dot_row(M, i, v, n) for i in g)
                    R = sum(r[i] for i in g)
                    if S <= 0 or S != S:
                        return None
                    m = (R / S) ** w
                    m = min(max(m, CLIP_LO), CLIP_HI)
                    for i in g:
                        u[i] = min(max(u[i] * m, CLIP_LO), CLIP_HI)
            else:
                for g in groups:
                    S = sum(v[j] * _dot_col(M, j, u, n) for j in g)
                    Ctot = sum(c[j] for j in g)
                    if S <= 0 or S != S:
                        return None
                    m = (Ctot / S) ** w
                    m = min(max(m, CLIP_LO), CLIP_HI)
                    for j in g:
                        v[j] = min(max(v[j] * m, CLIP_LO), CLIP_HI)
        else:
            return None
    for x in u + v:
        if x != x or x in (float("inf"), float("-inf")):
            return None
    s_row = [u[i] * _dot_row(M, i, v, n) for i in range(n)]
    s_col = [v[j] * _dot_col(M, j, u, n) for j in range(n)]
    for x in s_row + s_col:
        if x != x or x in (float("inf"), float("-inf")):
            return None
    e_row = sum(abs(s_row[i] - r[i]) / r[i] for i in range(n)) / n
    e_col = sum(abs(s_col[j] - c[j]) / c[j] for j in range(n)) / n
    imb = 0.5 * (e_row + e_col)
    if imb != imb or imb in (float("inf"), float("-inf")) or imb < 0:
        return None
    return imb, u, v


def _dense_block_matrix(rng, block_sizes, eps):
    n = sum(block_sizes)
    blk = []
    for g, sz in enumerate(block_sizes):
        blk += [g] * sz
    M0 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if blk[i] == blk[j]:
                M0[i][j] = round(rng.uniform(0.5, 1.5), 6)
            else:
                M0[i][j] = round(eps * rng.uniform(0.5, 1.5), 9)
    perm = list(range(n))
    rng.shuffle(perm)
    M = [[M0[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    newblk = [blk[perm[i]] for i in range(n)]
    return M, newblk


def _dense_mixed_matrix(rng, n):
    return [[round(rng.uniform(0.5, 1.5), 6) for _ in range(n)] for _ in range(n)]


def _targets_block(rng, n, newblk, k):
    skew = [rng.uniform(0.4, 2.2) for _ in range(k)]
    r = [skew[newblk[i]] * rng.uniform(0.6, 1.4) for i in range(n)]
    sr = sum(r)
    r = [x * n / sr for x in r]
    skew2 = [rng.uniform(0.4, 2.2) for _ in range(k)]
    c = [skew2[newblk[j]] * rng.uniform(0.6, 1.4) for j in range(n)]
    sc = sum(c)
    # rescale c to match sum(r) EXACTLY (both target n, float-precision only)
    # so a zero-residual scaling is genuinely reachable in the limit -- do
    # not round r/c afterward: rounding independently reintroduces a
    # sum(r) != sum(c) mismatch and an artificial convergence floor.
    c = [x * n / sc for x in c]
    return r, c


def _targets_mixed(rng, n):
    r = [rng.uniform(0.5, 1.5) for _ in range(n)]
    sr = sum(r)
    r = [x * n / sr for x in r]
    c = [rng.uniform(0.5, 1.5) for _ in range(n)]
    sc = sum(c)
    c = [x * n / sc for x in c]
    return r, c


def make_instances():
    # (kind, block_sizes|n, eps, budget)
    specs = [
        ("block", [10, 12], 1.0e-3, 30),
        ("block", [8, 9, 10], 5.0e-4, 34),
        ("block", [14, 16], 2.0e-3, 28),
        ("mixed", 30, None, 30),
        ("mixed", 34, None, 30),
        ("block", [6, 6, 7, 7], 1.0e-3, 40),
        ("mixed", 40, None, 34),
        ("block", [9, 10, 13], 3.0e-4, 50),
        ("mixed", 45, None, 40),
        ("block", [8, 32], 1.0e-3, 36),
    ]
    out = []
    for si, (kind, sz, eps, budget) in enumerate(specs):
        rng = random.Random(31000 + si * 97)
        if kind == "block":
            M, newblk = _dense_block_matrix(rng, sz, eps)
            n = sum(sz)
            r, c = _targets_block(rng, n, newblk, len(sz))
        else:
            n = sz
            M = _dense_mixed_matrix(rng, n)
            r, c = _targets_mixed(rng, n)
        pub = {"n": n, "budget": budget, "M": M, "r": r, "c": c}
        out.append({"public": pub, "hidden": {}})
    return out


def _validate_ops(pub, ans):
    n = pub["n"]; budget = pub["budget"]
    if not isinstance(ans, dict) or "ops" not in ans:
        return None
    ops = ans["ops"]
    if not isinstance(ops, list) or len(ops) > budget:
        return None
    clean = []
    for op in ops:
        if not isinstance(op, dict) or "type" not in op:
            return None
        t = op.get("type")
        if t not in ("row", "col", "block"):
            return None
        w = op.get("omega", None)
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            return None
        w = float(w)
        if w != w or w < OMEGA_LO - 1e-9 or w > OMEGA_HI + 1e-9:
            return None
        if t == "block":
            axis = op.get("axis")
            if axis not in ("row", "col"):
                return None
            groups = op.get("groups")
            if not isinstance(groups, list) or not groups:
                return None
            seen = [False] * n
            for g in groups:
                if not isinstance(g, list) or not g:
                    return None
                for idx in g:
                    if not isinstance(idx, int) or isinstance(idx, bool):
                        return None
                    if idx < 0 or idx >= n or seen[idx]:
                        return None
                    seen[idx] = True
            if not all(seen):
                return None
            clean.append({"type": "block", "axis": axis, "omega": w, "groups": groups})
        else:
            clean.append({"type": t, "omega": w})
    return clean


def baseline(inst):
    pub = inst["public"]
    budget = pub["budget"]
    ops = []
    while len(ops) < budget:
        ops.append({"type": "row", "omega": 1.0})
        if len(ops) < budget:
            ops.append({"type": "col", "omega": 1.0})
    res = replay(pub, ops)
    if res is None:
        return 1.0  # degenerate fallback, should not happen with valid instances
    return res[0]


def score(inst, ans):
    pub = inst["public"]
    ops = _validate_ops(pub, ans)
    if ops is None:
        return False, 0.0
    res = replay(pub, ops)
    if res is None:
        return False, 0.0
    imb = res[0]
    if imb != imb or imb < 0:
        return False, 0.0
    return True, imb


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
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        # NOTE: obj can legitimately converge to near machine precision
        # (a near-perfectly balanced instance within budget); floor only
        # guards the literal obj==0 division, must not exceed realistic
        # obj magnitudes or it would artificially SHRINK the ratio instead
        # of letting it saturate toward the min(1, ...) cap.
        r = min(1.0, 0.1 * b / obj) if obj > 0.0 else 1.0
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
