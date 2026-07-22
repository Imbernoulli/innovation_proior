# TIER: greedy
# The obvious first attempt: plain WalkSAT over ALL variables, uniformly, with no notion
# that some variables might matter more than others. It reads the weighted clauses, does
# greedy break/make flips with a little random noise to escape plateaus, and restarts from
# a fresh random full assignment a few times, keeping the best it ever saw. This handles
# the bulk of the (numerous, light) noise clauses just fine -- but a chunk of those light
# clauses are cheapest to satisfy when a handful of "important" variables sit at one
# particular value, and since this solver never distinguishes those variables from anyone
# else, it drifts them there right along with everything else, paying for it on the much
# heavier -- but rare -- clauses that need those same variables at the OPPOSITE value.
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


seed = (n * 1315423911 + m * 2654435761 + int(sum(weights) * 1000)) & 0x7FFFFFFF
rng = _rng(seed if seed > 0 else 1)

var_clauses = [[] for _ in range(n)]
for ci, c in enumerate(clauses):
    for lit in c:
        v = abs(lit) - 1
        sign = 1 if lit > 0 else -1
        var_clauses[v].append((ci, sign))


def lit_true(sign, val):
    return (sign > 0 and val == 1) or (sign < 0 and val == 0)


def solve_once(pool, max_flips):
    assign = [rng(0, 1) for _ in range(n)]
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

    for _ in range(max_flips):
        if not unsat_list:
            break
        ci = unsat_list[rng(0, len(unsat_list) - 1)]
        cand_vars = [abs(lit) - 1 for lit in clauses[ci] if (abs(lit) - 1) in pool]
        if not cand_vars:
            continue
        if rng(0, 9) < 3:
            v = cand_vars[rng(0, len(cand_vars) - 1)]
        else:
            best_gain = None
            best_vs = []
            for v in cand_vars:
                brk = 0.0
                mk = 0.0
                for (cj, sign) in var_clauses[v]:
                    if num_true[cj] == 1 and lit_true(sign, assign[v]):
                        brk += weights[cj]
                    elif not satisfied[cj] and lit_true(sign, 1 - assign[v]):
                        mk += weights[cj]
                gain = mk - brk
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_vs = [v]
                elif gain == best_gain:
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

        if total > best_total:
            best_total = total
            best_assign = list(assign)

    return best_total, best_assign


pool = set(range(n))
overall_best_total = None
overall_best_assign = None
RESTARTS = 4
FLIPS = 350
for _ in range(RESTARTS):
    t, a = solve_once(pool, FLIPS)
    if overall_best_total is None or t > overall_best_total:
        overall_best_total = t
        overall_best_assign = a

print(json.dumps({"assign": overall_best_assign}))
