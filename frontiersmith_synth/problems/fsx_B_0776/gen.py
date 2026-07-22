#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE hydraulic-trunk-sizing instance to stdout.
Deterministic: all randomness seeded from testId only.

Also performs an OFFLINE calibration search (over the pipe budget C) so that, for
THIS instance, a reasonably-informed allocator (a local proxy for 'strong') lands in a
target score band relative to the checker's own uniform-diameter baseline -- this keeps
every test case genuinely open (no case where the naive baseline already wins, and no
case where any allocator can trivially reach a perfect score / saturate the ratio cap).
The search only picks C; it is never shown to the solver beyond the final printed value.
"""
import random, sys

DIAMS = list(range(1, 9))  # 8 discrete diameter options, dimensionless units 1..8


def gen_topology(rng, n, testId):
    parent = [-1] * n  # node 0 = source; nodes 1..n-1 need demand
    kind = testId % 3
    if kind == 0:  # twin-bush trunk trap: a long trunk with TWO separate bushy
        # sub-networks branching off it at different depths, so budget must be
        # split between two competing trunk-investment needs, not just one.
        trunk_len = max(2, n // 4)
        prev = 0
        trunk_nodes = [0]
        for i in range(1, trunk_len + 1):
            parent[i] = prev
            prev = i
            trunk_nodes.append(i)
        mid = trunk_nodes[len(trunk_nodes) // 2]
        end = trunk_nodes[-1]
        rest = list(range(trunk_len + 1, n))
        half = len(rest) // 2
        pool_a = [mid]
        pool_b = [end]
        for i in rest[:half]:
            parent[i] = rng.choice(pool_a)
            pool_a.append(i)
        for i in rest[half:]:
            parent[i] = rng.choice(pool_b)
            pool_b.append(i)
    elif kind == 1:  # heavy-branch star trap: the source has a handful of direct
        # branches; one of them silently accumulates almost the whole rest of the
        # network behind it (huge downstream flow hiding behind a single edge).
        k = min(4, n - 1)
        direct = rng.sample(range(1, n), k) if n - 1 >= k else list(range(1, n))
        for d in direct:
            parent[d] = 0
        remaining = [v for v in range(1, n) if parent[v] == -1]
        pool = [direct[0]] + direct[1:2]
        for v in remaining:
            par = rng.choice(pool)
            parent[v] = par
            pool.append(v)
    else:  # general random attachment tree (non-adversarial control case)
        for v in range(1, n):
            parent[v] = rng.randint(0, v - 1)
    return parent


def gen_instance(testId):
    rng = random.Random(1_000_003 + testId * 97)
    sizes = [5, 8, 12, 18, 25, 32, 40, 50, 65, 80]
    n = sizes[testId - 1]
    parent = gen_topology(rng, n, testId)

    demand = [0] * n
    length = [0] * n
    unit_cost = [0] * n
    K = [0.0] * n
    for v in range(1, n):
        demand[v] = rng.randint(1, 6)
        length[v] = rng.randint(5, 20)
        unit_cost[v] = rng.randint(1, 3)
        K[v] = round(rng.uniform(0.5, 1.5), 6)
    return dict(n=n, parent=parent, demand=demand, length=length,
                unit_cost=unit_cost, K=K)


def children_of(inst):
    n = inst['n']; parent = inst['parent']
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    return children


def subtree_flow(inst, children):
    n = inst['n']
    order = []
    st = [(0, False)]
    while st:
        node, processed = st.pop()
        if processed:
            order.append(node); continue
        st.append((node, True))
        for c in children[node]:
            st.append((c, False))
    S = [0] * n
    for v in order:
        S[v] = inst['demand'][v] + sum(S[c] for c in children[v])
    return S


def edge_cost(inst, v, D):
    return inst['unit_cost'][v] * inst['length'][v] * D * D


def uniform_total_cost(inst, D):
    return sum(edge_cost(inst, v, D) for v in range(1, inst['n']))


def head_loss(inst, v, S, D):
    Q = S[v]
    return inst['K'][v] * inst['length'][v] * Q * Q / (D ** 5)


def path_loss_uniform(inst, S, children, D_const):
    n = inst['n']
    L = [0.0] * n
    st = [0]
    while st:
        v = st.pop()
        for c in children[v]:
            L[c] = L[v] + head_loss(inst, c, S, D_const)
            st.append(c)
    return L


def path_loss_actual(inst, S, children, diam_idx):
    n = inst['n']
    L = [0.0] * n
    st = [0]
    while st:
        v = st.pop()
        for c in children[v]:
            L[c] = L[v] + head_loss(inst, c, S, DIAMS[diam_idx[c]])
            st.append(c)
    return L


def total_unmet(inst, diam_idx, S, children, bestL, worstL):
    n = inst['n']
    L = path_loss_actual(inst, S, children, diam_idx)
    unmet = 0.0
    for v in range(1, n):
        denom = worstL[v] - bestL[v]
        if denom < 1e-9:
            f = 1.0
        else:
            f = (worstL[v] - L[v]) / denom
            f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        unmet += (1 - f) * inst['demand'][v]
    return unmet


def uniform_max_index(inst, C):
    best = 0
    for i, D in enumerate(DIAMS):
        if uniform_total_cost(inst, D) <= C:
            best = i
    return best


def calib_trivial(inst, C):
    return [uniform_max_index(inst, C)] * inst['n']


def calib_greedy(inst, C, S):
    """Proxy for the 'linear demand-share' recipe: water-fill the remaining budget
    proportional to each edge's own flow Q_e (a plausible but LINEAR heuristic)."""
    n = inst['n']
    idx = [0] * n
    remaining = C - uniform_total_cost(inst, DIAMS[0])
    active = [v for v in range(1, n) if idx[v] + 1 < len(DIAMS)]
    while remaining > 1e-9 and active:
        total_q = sum(S[v] for v in active) or 1
        bought_any = False
        still_active = []
        for v in active:
            share = remaining * S[v] / total_q
            c_next = edge_cost(inst, v, DIAMS[idx[v] + 1]) - edge_cost(inst, v, DIAMS[idx[v]])
            if c_next <= share + 1e-9 and c_next <= remaining:
                idx[v] += 1
                remaining -= c_next
                bought_any = True
            if idx[v] + 1 < len(DIAMS):
                still_active.append(v)
        active = still_active
        if not bought_any:
            break
    return idx


def subtree_lists(inst, children):
    n = inst['n']
    lists = [None] * n

    def collect(v):
        acc = [v]
        for c in children[v]:
            acc.extend(collect(c))
        lists[v] = acc
        return acc

    collect(0)
    return lists


def calib_strong(inst, C, S, children, bestL, worstL):
    """Proxy for an exact marginal-value greedy (used only to CALIBRATE the budget)."""
    n = inst['n']
    idx = [0] * n
    sub_nodes = subtree_lists(inst, children)
    denom = [max(1e-9, worstL[v] - bestL[v]) for v in range(n)]
    L = path_loss_actual(inst, S, children, idx)
    remaining = C - uniform_total_cost(inst, DIAMS[0])

    def f_of(v, Lv):
        x = (worstL[v] - Lv) / denom[v]
        return 0.0 if x < 0 else (1.0 if x > 1 else x)

    while remaining > 1e-9:
        best_v = None; best_score = -1.0; best_cost = None; best_dHL = None
        for v in range(1, n):
            if idx[v] + 1 >= len(DIAMS):
                continue
            D0, D1 = DIAMS[idx[v]], DIAMS[idx[v] + 1]
            Q = S[v]
            HL0 = inst['K'][v] * inst['length'][v] * Q * Q / (D0 ** 5)
            HL1 = inst['K'][v] * inst['length'][v] * Q * Q / (D1 ** 5)
            dHL = HL0 - HL1
            c_next = edge_cost(inst, v, D1) - edge_cost(inst, v, D0)
            if c_next > remaining or c_next <= 0:
                continue
            benefit = 0.0
            for u in sub_nodes[v]:
                if u == 0:
                    continue
                benefit += (f_of(u, L[u] - dHL) - f_of(u, L[u])) * inst['demand'][u]
            score = benefit / c_next
            if score > best_score:
                best_score = score; best_v = v; best_cost = c_next; best_dHL = dHL
        if best_v is None or best_score <= 1e-12:
            break
        idx[best_v] += 1
        remaining -= best_cost
        for u in sub_nodes[best_v]:
            L[u] -= best_dHL
    return idx


def calibrate_budget(inst):
    n = inst['n']
    children = children_of(inst)
    S = subtree_flow(inst, children)
    bestL = path_loss_uniform(inst, S, children, DIAMS[-1])
    worstL = path_loss_uniform(inst, S, children, DIAMS[0])
    cmin = uniform_total_cost(inst, DIAMS[0])
    cmax = uniform_total_cost(inst, DIAMS[-1])
    grid = sorted(set([x / 10000.0 for x in range(2, 201, 3)] +
                       [x / 1000.0 for x in range(21, 451, 5)]))
    lo, hi = 0.35, 0.90
    in_band = []
    all_cands = []
    for frac in grid:
        C = int(round(cmin + frac * (cmax - cmin)))
        if C < cmin:
            C = cmin
        tri = calib_trivial(inst, C)
        gre = calib_greedy(inst, C, S)
        stg = calib_strong(inst, C, S, children, bestL, worstL)
        B = total_unmet(inst, tri, S, children, bestL, worstL)
        Fg = total_unmet(inst, gre, S, children, bestL, worstL)
        Fs = total_unmet(inst, stg, S, children, bestL, worstL)
        rS = min(1000.0, 100.0 * B / max(1e-9, Fs)) / 1000.0
        rG = min(1000.0, 100.0 * B / max(1e-9, Fg)) / 1000.0
        rec = (frac, C, rS, rG)
        all_cands.append(rec)
        if lo <= rS <= hi:
            in_band.append(rec)
    pool = in_band if in_band else all_cands
    strict = [r for r in pool if r[2] - r[3] >= 0.15]
    if strict:
        best = max(strict, key=lambda r: r[3])
    else:
        best = max(pool, key=lambda r: (r[2] - r[3]))
    return best[1]  # calibrated C


def main():
    testId = int(sys.argv[1])
    inst = gen_instance(testId)
    n = inst['n']
    C = calibrate_budget(inst)
    out = []
    out.append(f"{n}")
    out.append(f"{len(DIAMS)} " + " ".join(str(d) for d in DIAMS))
    out.append(f"{C}")
    for v in range(1, n):
        out.append(f"{inst['parent'][v] + 1} {inst['demand'][v]} {inst['length'][v]} "
                    f"{inst['unit_cost'][v]} {inst['K'][v]:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
