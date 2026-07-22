# TIER: strong
"""
Insight: reformulate as an exact search over (candidate patient set, set of
tests already prepared along this path) rather than a greedy one-step
information-gain ranking. At each node we try EVERY not-yet-used test, price
it by the MARGINAL number of NEW prep instructions it needs given what is
already "live" on this path (shared prefixes make some expensive-looking
tests nearly free once a sibling has run, and some individually-informative
tests can be a trap: informative alone, yet totally redundant once cheaper
tests are known), and recurse. Memoized by (frozenset(patient ids), frozenset
of used tests) -- the state space stays small because tests are few. This
finds the population's true minimum expected-cost decision program, whether
or not it matches the entropy-greedy order.
"""
import sys
from fractions import Fraction


def resolve(tok, F, vals):
    if tok[0] == 'F':
        return F[int(tok[1:])]
    if tok[0] == 'I':
        return vals[int(tok[1:])]
    return int(tok[1:])


def eval_all(instrs, F):
    vals = []
    for (op, a, b) in instrs:
        va, vb = resolve(a, F, vals), resolve(b, F, vals)
        if op == 'ADD':
            v = va + vb
        elif op == 'SUB':
            v = va - vb
        else:
            v = va * vb
        vals.append(v)
    return vals


def closure_of(instrs, final_idx):
    seen = set()
    stack = [final_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        op, a, b = instrs[idx]
        for tok in (a, b):
            if tok[0] == 'I':
                j = int(tok[1:])
                if j not in seen:
                    stack.append(j)
    return seen


def main():
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v

    K = int(nxt()); M = int(nxt()); T = int(nxt()); N = int(nxt())
    instrs = []
    for _ in range(M):
        instrs.append((nxt(), nxt(), nxt()))
    tests = []
    for _ in range(T):
        tests.append((int(nxt()), int(nxt())))
    patients = []
    for i in range(N):
        F = [int(nxt()) for _ in range(K)]
        w = int(nxt()); lab = int(nxt())
        patients.append((F, w, lab))

    test_closure = [closure_of(instrs, fin) for (fin, thr) in tests]

    outcomes = []
    labels = []
    weights = []
    for (F, w, lab) in patients:
        vals = eval_all(instrs, F)
        bits = tuple(1 if vals[fin] >= thr else 0 for (fin, thr) in tests)
        outcomes.append(bits)
        labels.append(lab)
        weights.append(w)

    # Collapse patients sharing an identical outcome-vector into one group
    # (they are always co-routed, and -- by the input's construction -- share
    # a label too). This keeps the DP's state space small and fast.
    groups = {}
    for i in range(N):
        groups.setdefault(outcomes[i], [0, None])
        groups[outcomes[i]][0] += weights[i]
        groups[outcomes[i]][1] = labels[i]
    group_keys = list(groups.keys())  # list of outcome-tuples
    group_w = [groups[k][0] for k in group_keys]
    group_lab = [groups[k][1] for k in group_keys]
    G = len(group_keys)

    memo = {}

    def solve(idxs, used, live):
        # idxs: tuple of group indices (sorted) still ambiguous at this node
        labs = set(group_lab[i] for i in idxs)
        if len(labs) == 1:
            return 0, ('LEAF', next(iter(labs)))
        key = (idxs, used)
        if key in memo:
            return memo[key]
        best_cost = None
        best_plan = None
        for t in range(T):
            if (used >> t) & 1:
                continue
            new_instrs = test_closure[t] - live
            cost_t = len(new_instrs) + 1
            live2 = live | test_closure[t]
            idxs0 = tuple(i for i in idxs if group_keys[i][t] == 0)
            idxs1 = tuple(i for i in idxs if group_keys[i][t] == 1)
            tot_w = sum(group_w[i] for i in idxs)
            sub_total = cost_t * tot_w
            if idxs0:
                c0, plan0 = solve(idxs0, used | (1 << t), frozenset(live2))
                sub_total += c0
            else:
                plan0 = None
            if idxs1:
                c1, plan1 = solve(idxs1, used | (1 << t), frozenset(live2))
                sub_total += c1
            else:
                plan1 = None
            if best_cost is None or sub_total < best_cost:
                best_cost = sub_total
                best_plan = ('TEST', t, idxs0, plan0, idxs1, plan1)
        memo[key] = (best_cost, best_plan)
        return memo[key]

    all_idxs = tuple(range(G))
    total_w = sum(group_w)
    _, plan = solve(all_idxs, 0, frozenset())

    # materialize `plan` (a recursive tuple structure) into flat node list,
    # re-deriving each TEST node's children plans on demand (memo makes this cheap).
    nodes = []

    def emit(idxs, used, live, plan_hint):
        if plan_hint[0] == 'LEAF':
            nid = len(nodes)
            nodes.append(('LEAF', plan_hint[1]))
            return nid
        _, t, idxs0, plan0, idxs1, plan1 = plan_hint
        nid = len(nodes)
        nodes.append(None)
        live2 = live | test_closure[t]
        if idxs0:
            lo = emit(idxs0, used | (1 << t), live2, plan0)
        else:
            # unreachable branch (no patient group takes it): emit a stub leaf
            lo = len(nodes)
            nodes.append(('LEAF', group_lab[idxs[0]]))
        if idxs1:
            hi = emit(idxs1, used | (1 << t), live2, plan1)
        else:
            hi = len(nodes)
            nodes.append(('LEAF', group_lab[idxs[0]]))
        nodes[nid] = ('TEST', t, lo, hi)
        return nid

    root = emit(all_idxs, 0, frozenset(), plan)
    assert root == 0

    out_lines = [str(len(nodes))]
    for nd in nodes:
        if nd[0] == 'LEAF':
            out_lines.append("LEAF %d" % nd[1])
        else:
            _, ti, lo, hi = nd
            out_lines.append("TEST %d %d %d" % (ti, lo, hi))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
