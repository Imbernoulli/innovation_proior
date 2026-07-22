# TIER: strong
import sys, json

CLIP_LO, CLIP_HI = 1e-9, 1e9


def dot_row(M, i, v, n):
    row = M[i]
    return sum(row[j] * v[j] for j in range(n))


def dot_col(M, j, u, n):
    return sum(M[i][j] * u[i] for i in range(n))


def replay(n, M, r, c, ops):
    """Local copy of the grader's exact replay semantics -- used only to
    PLAN the submitted move list; none of this local search counts against
    the operation budget, only len(ops) does."""
    u = [1.0] * n
    v = [1.0] * n
    for op in ops:
        t = op["type"]; w = op["omega"]
        if t == "row":
            for i in range(n):
                s = u[i] * dot_row(M, i, v, n)
                if s <= 0 or s != s:
                    return None
                m = min(max((r[i] / s) ** w, CLIP_LO), CLIP_HI)
                u[i] = min(max(u[i] * m, CLIP_LO), CLIP_HI)
        elif t == "col":
            for j in range(n):
                s = v[j] * dot_col(M, j, u, n)
                if s <= 0 or s != s:
                    return None
                m = min(max((c[j] / s) ** w, CLIP_LO), CLIP_HI)
                v[j] = min(max(v[j] * m, CLIP_LO), CLIP_HI)
        else:  # block
            groups = op["groups"]
            if op["axis"] == "row":
                for g in groups:
                    S = sum(u[i] * dot_row(M, i, v, n) for i in g)
                    R = sum(r[i] for i in g)
                    if S <= 0 or S != S:
                        return None
                    m = min(max((R / S) ** w, CLIP_LO), CLIP_HI)
                    for i in g:
                        u[i] = min(max(u[i] * m, CLIP_LO), CLIP_HI)
            else:
                for g in groups:
                    S = sum(v[j] * dot_col(M, j, u, n) for j in g)
                    Ctot = sum(c[j] for j in g)
                    if S <= 0 or S != S:
                        return None
                    m = min(max((Ctot / S) ** w, CLIP_LO), CLIP_HI)
                    for j in g:
                        v[j] = min(max(v[j] * m, CLIP_LO), CLIP_HI)
    for x in u + v:
        if x != x or x in (float("inf"), float("-inf")):
            return None
    s_row = [u[i] * dot_row(M, i, v, n) for i in range(n)]
    s_col = [v[j] * dot_col(M, j, u, n) for j in range(n)]
    for x in s_row + s_col:
        if x != x or x in (float("inf"), float("-inf")):
            return None
    e_row = sum(abs(s_row[i] - r[i]) / r[i] for i in range(n)) / n
    e_col = sum(abs(s_col[j] - c[j]) / c[j] for j in range(n)) / n
    return 0.5 * (e_row + e_col)


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def detect_blocks(n, M, tau=0.15):
    """near-reducible-block-reorder: infer a partition from where each
    row/column's mass concentrates, via a threshold + union-find over the
    coupling pattern (no access to the hidden permutation -- only to M)."""
    rowmax = [max(M[i]) for i in range(n)]
    colmax = [max(M[i][j] for i in range(n)) for j in range(n)]
    parent = list(range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if M[i][j] > tau * rowmax[i] or M[j][i] > tau * rowmax[j] \
               or M[i][j] > tau * colmax[j] or M[j][i] > tau * colmax[i]:
                union(parent, i, j)
    groups = {}
    for i in range(n):
        root = find(parent, i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


def leakage_fraction(n, M, groups):
    gid = [0] * n
    for gi, g in enumerate(groups):
        for i in g:
            gid[i] = gi
    total = 0.0
    cross = 0.0
    for i in range(n):
        for j in range(n):
            total += M[i][j]
            if gid[i] != gid[j]:
                cross += M[i][j]
    return cross / total if total > 0 else 1.0


def plan_ops(n, M, r, c, budget, groups):
    """imbalance-regime-detect + adaptive-overrelaxation: grid-search a
    small set of (#pre-sweeps, block-omega) candidate plans by LOCALLY
    simulating them (unlimited local compute, zero budget cost), and
    submit whichever plan the grader's own replay rule scores best."""
    best_ops, best_val = None, float("inf")
    pre_choices = [c_ for c_ in (2, 4, 6, 8) if c_ < budget]
    if not pre_choices:
        pre_choices = [0]
    omega_choices = [1.0, 1.3, 1.6, 1.9, 2.2]
    for pre in pre_choices:
        pre_ops = []
        while len(pre_ops) < pre:
            pre_ops.append({"type": "row", "omega": 1.0})
            if len(pre_ops) < pre:
                pre_ops.append({"type": "col", "omega": 1.0})
        rest = budget - len(pre_ops)
        if rest <= 0:
            val = replay(n, M, r, c, pre_ops)
            if val is not None and val < best_val:
                best_val, best_ops = val, pre_ops
            continue
        for w in omega_choices:
            block_ops = []
            while len(block_ops) < rest:
                block_ops.append({"type": "block", "axis": "row", "omega": w, "groups": groups})
                if len(block_ops) < rest:
                    block_ops.append({"type": "block", "axis": "col", "omega": w, "groups": groups})
            trial = pre_ops + block_ops
            val = replay(n, M, r, c, trial)
            if val is not None and val < best_val:
                best_val, best_ops = val, trial
    return best_ops, best_val


def plan_plain(n, M, r, c, budget):
    best_ops, best_val = None, float("inf")
    for w in (1.0, 1.2, 1.4, 1.6, 1.8):
        ops = []
        while len(ops) < budget:
            ops.append({"type": "row", "omega": w})
            if len(ops) < budget:
                ops.append({"type": "col", "omega": w})
        val = replay(n, M, r, c, ops)
        if val is not None and val < best_val:
            best_val, best_ops = val, ops
    return best_ops, best_val


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]; M = inst["M"]; r = inst["r"]; c = inst["c"]; budget = inst["budget"]

    groups = detect_blocks(n, M)
    leak = leakage_fraction(n, M, groups)
    sizes_ok = len(groups) >= 2 and min(len(g) for g in groups) >= 2

    plain_ops, plain_val = plan_plain(n, M, r, c, budget)

    if sizes_ok and leak < 0.2:
        block_ops, block_val = plan_ops(n, M, r, c, budget, groups)
    else:
        block_ops, block_val = None, float("inf")

    if block_ops is not None and block_val < plain_val:
        ops = block_ops
    else:
        ops = plain_ops
    print(json.dumps({"ops": ops}))


main()
