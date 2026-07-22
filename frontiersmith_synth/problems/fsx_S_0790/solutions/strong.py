# TIER: strong
# THE INSIGHT: don't treat every variable the same. First PROBE for a backbone by running
# unit propagation restricted to only the heaviest clauses (found via the biggest gap in
# the sorted weight spectrum -- no hardcoded constant). Heavy unit clauses pin a variable
# directly; heavy 2-literal clauses are implications, and propagating them to closure (
# standard 2-SAT-style forcing) recovers the value of every variable those heavy clauses
# touch -- for free, with certainty, since those clauses vastly outweigh anything else that
# could pull those variables around. FREEZE those variables at their forced value. Only
# THEN run a focused WalkSAT -- flips restricted to the remaining free variables, with a
# short tabu list so it doesn't immediately undo its own recent flips, plus periodic
# restarts of just the free assignment (the frozen part never moves) -- to mop up the
# numerous lighter, noisier clauses among what's left.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
clauses = inst["clauses"]
weights = inst["weights"]
m = len(clauses)


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


seed = (n * 2971215073 + m * 1597334677 + int(sum(weights) * 1000)) & 0x7FFFFFFF
rng = _rng(seed if seed > 0 else 1)

# ---- 1. probe the backbone via the biggest gap in the weight spectrum ----
ws = sorted(set(weights))
if len(ws) > 1:
    best_gap = -1.0
    thresh = ws[-1]
    for i in range(len(ws) - 1):
        g = ws[i + 1] - ws[i]
        if g > best_gap:
            best_gap = g
            thresh = (ws[i] + ws[i + 1]) / 2.0
else:
    thresh = ws[0] + 1.0

heavy = [c for c, w in zip(clauses, weights) if w >= thresh]

forced = {}
for c in heavy:
    if len(c) == 1:
        lit = c[0]
        v = abs(lit) - 1
        val = 1 if lit > 0 else 0
        if v not in forced:
            forced[v] = val

changed = True
guard = 0
while changed and guard < len(heavy) + 5:
    changed = False
    guard += 1
    for c in heavy:
        if len(c) != 2:
            continue
        l1, l2 = c
        v1, v2 = abs(l1) - 1, abs(l2) - 1
        if v1 in forced:
            v1_true = (forced[v1] == 1 and l1 > 0) or (forced[v1] == 0 and l1 < 0)
            if not v1_true and v2 not in forced:
                forced[v2] = 1 if l2 > 0 else 0
                changed = True
        if v2 in forced:
            v2_true = (forced[v2] == 1 and l2 > 0) or (forced[v2] == 0 and l2 < 0)
            if not v2_true and v1 not in forced:
                forced[v1] = 1 if l1 > 0 else 0
                changed = True

free_vars = [v for v in range(n) if v not in forced]
pool = set(free_vars)

# ---- 2. focused-flip WalkSAT + tabu, restricted to the free set, with restarts ----
var_clauses = [[] for _ in range(n)]
for ci, c in enumerate(clauses):
    for lit in c:
        v = abs(lit) - 1
        sign = 1 if lit > 0 else -1
        var_clauses[v].append((ci, sign))


def lit_true(sign, val):
    return (sign > 0 and val == 1) or (sign < 0 and val == 0)


def init_assign():
    a = [0] * n
    for v, val in forced.items():
        a[v] = val
    for v in free_vars:
        a[v] = rng(0, 1)
    return a


def solve_once(max_flips, tabu_tenure):
    assign = init_assign()
    num_true = [0] * m
    satisfied = [False] * m
    total = 0.0
    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            if lit_true(sign, assign[v]):
                cnt += 1
        num_true[ci] = cnt
        satisfied[ci] = cnt > 0
        if satisfied[ci]:
            total += weights[ci]

    unsat_pos = {}
    unsat_list = []
    for ci in range(m):
        if not satisfied[ci]:
            unsat_pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def unsat_remove(ci):
        p = unsat_pos.pop(ci)
        last = unsat_list.pop()
        if p < len(unsat_list):
            unsat_list[p] = last
            unsat_pos[last] = p

    def unsat_add(ci):
        unsat_pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    best_total = total
    best_assign = list(assign)
    tabu_until = {}

    for it in range(max_flips):
        if not unsat_list:
            break
        ci = unsat_list[rng(0, len(unsat_list) - 1)]
        cand_vars = [abs(lit) - 1 for lit in clauses[ci] if (abs(lit) - 1) in pool]
        if not cand_vars:
            continue

        def score(v):
            brk = 0.0
            mk = 0.0
            for (cj, sign) in var_clauses[v]:
                if num_true[cj] == 1 and lit_true(sign, assign[v]):
                    brk += weights[cj]
                elif not satisfied[cj] and lit_true(sign, 1 - assign[v]):
                    mk += weights[cj]
            return mk - brk

        if rng(0, 9) < 2:
            v = cand_vars[rng(0, len(cand_vars) - 1)]
        else:
            allowed = [v for v in cand_vars if tabu_until.get(v, -1) <= it]
            pick_from = allowed if allowed else cand_vars
            best_gain = None
            best_vs = []
            for v in pick_from:
                g = score(v)
                if best_gain is None or g > best_gain:
                    best_gain = g
                    best_vs = [v]
                elif g == best_gain:
                    best_vs.append(v)
            v = best_vs[rng(0, len(best_vs) - 1)]

        for (cj, sign) in var_clauses[v]:
            was_true = lit_true(sign, assign[v])
            if was_true:
                num_true[cj] -= 1
                if num_true[cj] == 0:
                    satisfied[cj] = False
                    total -= weights[cj]
                    unsat_add(cj)
            else:
                num_true[cj] += 1
                if num_true[cj] == 1:
                    satisfied[cj] = True
                    total += weights[cj]
                    unsat_remove(cj)
        assign[v] = 1 - assign[v]
        tabu_until[v] = it + tabu_tenure

        if total > best_total:
            best_total = total
            best_assign = list(assign)

    return best_total, best_assign


overall_best_total = None
overall_best_assign = None
RESTARTS = 4
FLIPS = 350
TABU = 7
for _ in range(RESTARTS):
    t, a = solve_once(FLIPS, TABU)
    if overall_best_total is None or t > overall_best_total:
        overall_best_total = t
        overall_best_assign = a

print(json.dumps({"assign": overall_best_assign}))
