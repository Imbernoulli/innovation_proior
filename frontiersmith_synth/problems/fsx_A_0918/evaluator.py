#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0918 -- "Rounding a Fractional Task-Assignment Plan Along Its
Hidden Cycles" (family: pipage-assignment-rounding; format B, quality-metric).

THEME.  N tasks must EACH be assigned to exactly one of M "desks" (agents). Desk k has an
integer capacity[k]; assigning task i to desk k consumes weight[i][k] units of that desk's
capacity (the weight DEPENDS ON BOTH the task and the desk -- the same task can be cheap on
one desk and expensive on another) and earns value[i][k]. Maximize total earned value. One
designated desk (always the LAST index) is a generous, low-value "overflow" option that is
always affordable, so a feasible assignment always exists -- the real decision is how to use
the tightly-capacitated desks well.

MECHANICS make this genuinely different from plain knapsack-style assignment: because
weight[i][k] varies with BOTH i and k (not just i), the natural LP relaxation of this
assignment problem (x[i][k] in [0,1], sum_k x[i][k] = 1 per task, sum_i weight[i][k]*x[i][k]
<= capacity[k] per desk) is NOT totally unimodular. Its optimal vertex can be genuinely
fractional, and -- because every task-row is a mandatory-total-1 equality while every desk-
column is only a capacity inequality -- the fractional support graph (edges with
0 < x[i][k] < 1) decomposes into simple PATHS and CYCLES that alternate task/desk/task/desk.
Some instances below are built so that two (or more) desks are each the best-looking choice
for two competing tasks at once: independently rounding each x[i][k] to its nearer integer,
or greedily grabbing whichever desk currently looks best task-by-task, locks in a task's
locally-fine choice that starves a *different* task of the desk it badly needed, landing far
below the value achievable by resolving the whole fractional cycle together.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"instance_id": str, "n_tasks": N, "n_agents": M,
             "capacity": [cap_0, ..., cap_{M-1}],
             "weight": [[w_{i,0}, ..., w_{i,M-1}] for i in range(N)],
             "value":  [[v_{i,0}, ..., v_{i,M-1}] for i in range(N)]}
  stdout: ONE JSON object: {"assign": [k_0, ..., k_{N-1}]}
          k_i: an INTEGER desk index in [0, M-1] -- the single desk task i is sent to.

  A valid answer has "assign" a list of exactly N integers, each in [0, M-1], such that for
  every desk k, sum of weight[i][k] over tasks i with assign[i]==k is <= capacity[k]. Any
  wrong length/type, out-of-range entry, capacity violation, a crash, timeout, or non-JSON
  output makes that instance score 0.0.

SCORING.  Per instance the evaluator computes, itself, two references NEVER revealed to the
candidate: obj_base (a naive first-fit-by-index construction that ignores value entirely) and
obj_ref (an internal LP-relaxation + cycle-pipage-rounding + value-improving-swap-repair
procedure -- a strong but not provably-optimal ceiling). With obj_ref > obj_base (higher
value is better):
    r = clamp(0.1 + 0.9 * (obj_cand - obj_base) / max(1e-9, 1.3*(obj_ref - obj_base)), 0, 1)
Matching the naive baseline scores ~0.1; matching the internal reference scores ~0.79; there
is headroom above that (the internal reference is a heuristic, not a proven optimum). The
reported Ratio is the mean r over 10 fixed instances; Vector lists the 10 scores.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance. obj_base/obj_ref are computed by
THIS parent process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, json
import isorun

TOL = 1e-6


# =========================== LP relaxation (mechanism 1) ============================
def _lp_relax(N, M, weight, capacity, value):
    from scipy.optimize import linprog
    nv = N * M
    c = [0.0] * nv
    for i in range(N):
        for k in range(M):
            # tiny deterministic index-based perturbation breaks LP ties so the optimal
            # vertex found is generic (each fractional task has exactly two fractional
            # desks), never revealed to / dependent on anything but fixed indices.
            c[i * M + k] = -(value[i][k] + 1e-6 * (i * M + k))
    A_eq, b_eq = [], []
    for i in range(N):
        row = [0.0] * nv
        for k in range(M):
            row[i * M + k] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    A_ub, b_ub = [], []
    for k in range(M):
        row = [0.0] * nv
        for i in range(N):
            row[i * M + k] = float(weight[i][k])
        A_ub.append(row)
        b_ub.append(float(capacity[k]))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=[(0.0, 1.0)] * nv, method="highs")
    if not res.success:
        return [[1.0 if k == M - 1 else 0.0 for k in range(M)] for _ in range(N)]
    xf = res.x
    return [[max(0.0, min(1.0, float(xf[i * M + k]))) for k in range(M)] for i in range(N)]


# ==================== pipage cycle rounding (mechanism 2) ============================
def _pipage_round(N, M, weight, value, capacity, x):
    assign = [None] * N
    frac = {}
    load = [0.0] * M  # running load of desks already fixed to an integral task
    for i in range(N):
        fk = [k for k in range(M) if TOL < x[i][k] < 1.0 - TOL]
        if fk:
            frac[i] = set(fk)
        else:
            k1 = max(range(M), key=lambda k: x[i][k])
            assign[i] = k1
            load[k1] += weight[i][k1]

    # defensive: collapse any task with >2 fractional desks (rare LP degeneracy) down to
    # its two highest-value fractional desks, so the cycle/leaf logic below (which assumes
    # exactly two fractional desks per fractional task -- the generic case for this LP's
    # extreme points) always has well-defined input.
    for i in list(frac.keys()):
        while len(frac[i]) > 2:
            keep = set(sorted(frac[i], key=lambda k: -value[i][k])[:2])
            drop = frac[i] - keep
            frac[i] = keep
            for k in drop:
                pass  # dropped below when agent_tasks is built

    agent_tasks = {}
    for i, ks in frac.items():
        for k in ks:
            agent_tasks.setdefault(k, set()).add(i)

    steps = 0
    while frac and steps <= N + 5:
        steps += 1
        # (a) leaf desk: exactly one fractional task still touches it -> peel: push that
        # task's whole unit-mass AWAY from the leaf desk, onto its other fractional desk.
        leaf_k = None
        for k, tasks in agent_tasks.items():
            if len(tasks) == 1:
                leaf_k = k
                break
        if leaf_k is not None:
            i = next(iter(agent_tasks[leaf_k]))
            other = [k for k in frac[i] if k != leaf_k]
            k2 = other[0] if other else leaf_k
            assign[i] = k2
            load[k2] += weight[i][k2]
            for k in list(frac[i]):
                agent_tasks[k].discard(i)
                if not agent_tasks[k]:
                    del agent_tasks[k]
            del frac[i]
            continue

        # (b) no leaf left: the remainder is a union of simple cycles. Trace one, 2-color
        # it (the two ways to walk the alternating cycle). Prefer whichever coloring keeps
        # every touched desk within its REMAINING capacity (given what earlier peels/cycles
        # already committed); if both (or neither) are capacity-feasible, break the tie by
        # whichever earns more value -- this is the "round along the fractional cycle two
        # variables at a time, preserving feasibility, value-improving" pipage step.
        i0 = next(iter(frac))
        ks0 = sorted(frac[i0])
        tasks_seq = [i0]
        agents_seq = [ks0[0]]
        cur_task, cur_agent = i0, ks0[0]
        while True:
            others = [t for t in agent_tasks[cur_agent] if t != cur_task]
            if not others:
                break
            nxt_task = others[0]
            if nxt_task == i0:
                break
            tasks_seq.append(nxt_task)
            ks = sorted(frac[nxt_task])
            nxt_agent = ks[0] if ks[0] != cur_agent else ks[1]
            agents_seq.append(nxt_agent)
            cur_task, cur_agent = nxt_task, nxt_agent
        L = len(tasks_seq)
        choiceA = {tasks_seq[j]: agents_seq[j] for j in range(L)}
        choiceB = {tasks_seq[j]: agents_seq[(j - 1) % L] for j in range(L)}

        def _feasible(choice):
            extra = {}
            for t in tasks_seq:
                k = choice[t]
                extra[k] = extra.get(k, 0.0) + weight[t][k]
            return all(load[k] + e <= capacity[k] + 1e-6 for k, e in extra.items())

        feasA, feasB = _feasible(choiceA), _feasible(choiceB)
        valA = sum(value[t][choiceA[t]] for t in tasks_seq)
        valB = sum(value[t][choiceB[t]] for t in tasks_seq)
        if feasA and not feasB:
            chosen = choiceA
        elif feasB and not feasA:
            chosen = choiceB
        else:
            chosen = choiceA if valA >= valB else choiceB
        for t in tasks_seq:
            assign[t] = chosen[t]
            load[chosen[t]] += weight[t][chosen[t]]
            for k in list(frac.get(t, ())):
                if k in agent_tasks:
                    agent_tasks[k].discard(t)
                    if not agent_tasks[k]:
                        del agent_tasks[k]
            frac.pop(t, None)

    # safety net: anything somehow left unresolved goes to its best-value fractional desk
    for i in list(frac.keys()):
        assign[i] = max(frac[i], key=lambda k: value[i][k])
    for i in range(N):
        if assign[i] is None:
            assign[i] = M - 1
    return assign


# ================= value-improving swap repair (mechanism 3) =========================
def _swap_repair(N, M, weight, capacity, value, assign, max_passes=200):
    assign = list(assign)
    overflow_idx = M - 1

    def loads():
        L = [0.0] * M
        for i in range(N):
            L[assign[i]] += weight[i][assign[i]]
        return L

    for _ in range(max_passes):
        load = loads()
        viol = [(load[k] - capacity[k], k) for k in range(M) if load[k] > capacity[k] + 1e-9]
        if not viol:
            break
        viol.sort(reverse=True)
        k_bad = viol[0][1]
        here = [i for i in range(N) if assign[i] == k_bad]

        best_move = None
        for i in here:
            for k2 in range(M):
                if k2 == k_bad:
                    continue
                if weight[i][k2] <= capacity[k2] - load[k2] + 1e-9:
                    score = value[i][k2] - value[i][k_bad]
                    if best_move is None or score > best_move[0]:
                        best_move = (score, i, k2)
        if best_move is not None:
            _, i, k2 = best_move
            assign[i] = k2
            continue

        best_swap = None
        for i in here:
            for j in range(N):
                if assign[j] == k_bad or j == i:
                    continue
                k2 = assign[j]
                new_k2 = load[k2] - weight[j][k2] + weight[i][k2]
                new_kbad = load[k_bad] - weight[i][k_bad] + weight[j][k_bad]
                if new_k2 <= capacity[k2] + 1e-9 and new_kbad <= capacity[k_bad] + 1e-9:
                    score = (value[i][k2] + value[j][k_bad]) - (value[i][k_bad] + value[j][k2])
                    if best_swap is None or score > best_swap[0]:
                        best_swap = (score, i, j, k2)
        if best_swap is not None:
            _, i, j, k2 = best_swap
            assign[i], assign[j] = k2, k_bad
            continue

        if here:
            i = min(here, key=lambda t: value[t][k_bad])
            assign[i] = overflow_idx
        else:
            break
    return assign


def _internal_reference(inst):
    N, M = inst["n_tasks"], inst["n_agents"]
    weight, value, capacity = inst["weight"], inst["value"], inst["capacity"]
    x = _lp_relax(N, M, weight, capacity, value)
    assign = _pipage_round(N, M, weight, value, capacity, x)
    assign = _swap_repair(N, M, weight, capacity, value, assign)
    return assign


def _naive_baseline(inst):
    # weakest possible feasible construction: everyone to the overflow desk (this is also
    # exactly what solutions/trivial.py does, so a correct trivial submission scores ~0.1).
    N, M = inst["n_tasks"], inst["n_agents"]
    return [M - 1] * N


def _value_of(inst, assign):
    return sum(inst["value"][i][assign[i]] for i in range(inst["n_tasks"]))


# =============================== instance construction ================================
GADGETS = {
    # (weight_ii, weight_io, weight_oi, weight_oo), (value_...), capacity  -- verified to be
    # a genuine fractional LP cycle (both tasks split between both desks) whose greedy
    # first-come resolution is far from optimal.
    "G1": ([[11, 10], [10, 12]], [[19, 18], [18, 20]], 11),
    "G2": ([[9, 8], [8, 10]], [[16, 15], [15, 17]], 9),
    "G3": ([[14, 12], [12, 15]], [[24, 22], [22, 26]], 14),
    "G4": ([[10, 9], [9, 11]], [[18, 16], [16, 20]], 10),
    "G5": ([[13, 11], [11, 14]], [[22, 20], [20, 23]], 13),
    "G6": ([[8, 7], [7, 9]], [[15, 13], [13, 16]], 8),
}
OVERFLOW_CAP = 500.0
OVERFLOW_W, OVERFLOW_V = 1.0, 2.0
BG_W_LO, BG_W_HI = 6, 9
BG_V_LO, BG_V_HI = 1, 4


def _lcg(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _build(name, seed, n_tasks, n_real_agents, gadget_placements, filler, cap_overrides=None):
    """
    gadget_placements: list of (gadget_key, task_i, task_j, agent_k, agent_l)
    filler: (task_list, agent_idx, value, weight, capacity)  or None
    cap_overrides: {agent_idx: capacity} for agents not set by a gadget/filler
    """
    M = n_real_agents + 1
    overflow = M - 1
    capacity = [0.0] * M
    weight = [[None] * M for _ in range(n_tasks)]
    value = [[None] * M for _ in range(n_tasks)]

    for gk, ti, tj, ak, al in gadget_placements:
        gw, gv, gcap = GADGETS[gk]
        capacity[ak] = float(gcap)
        capacity[al] = float(gcap)
        weight[ti][ak], weight[ti][al] = float(gw[0][0]), float(gw[0][1])
        weight[tj][ak], weight[tj][al] = float(gw[1][0]), float(gw[1][1])
        value[ti][ak], value[ti][al] = float(gv[0][0]), float(gv[0][1])
        value[tj][ak], value[tj][al] = float(gv[1][0]), float(gv[1][1])

    if filler is not None:
        tlist, fa, fv, fw, fcap = filler
        capacity[fa] = float(fcap)
        for t in tlist:
            weight[t][fa] = float(fw)
            value[t][fa] = float(fv)

    if cap_overrides:
        for a, c in cap_overrides.items():
            capacity[a] = float(c)

    capacity[overflow] = OVERFLOW_CAP
    for i in range(n_tasks):
        weight[i][overflow] = OVERFLOW_W
        value[i][overflow] = OVERFLOW_V

    rng = _lcg(seed)
    for i in range(n_tasks):
        for k in range(M):
            if weight[i][k] is None:
                weight[i][k] = float(rng(BG_W_LO, BG_W_HI))
                value[i][k] = float(rng(BG_V_LO, BG_V_HI))
            if capacity[k] == 0.0:
                capacity[k] = float(rng(BG_W_LO, BG_W_HI) * 3)

    return {"name": name, "n_tasks": n_tasks, "n_agents": M,
            "capacity": capacity, "weight": weight, "value": value}


def _build_instances():
    out = []
    # filler capacity is deliberately tight: room for all-but-one of its tasks at the
    # dedicated weight, so a correct solver must route exactly one task elsewhere (a small,
    # cheap decision) -- and a candidate that ignores capacity when picking each task's
    # single best desk (the "invalid" tier) reliably overflows it.
    def fcap(tasks, w):
        wi = int(w)
        n = len(tasks)
        return float(wi * (n - 1) + (wi + 1) // 2)

    # -- 0,1,2: calm -- every task has one clearly dominant desk; only a single spillover
    # decision is needed, no competing-cycle structure.
    out.append(_build("c0", 1101, 4, 3, [], ([0, 1, 2, 3], 0, 12.0, 4.0, fcap([0, 1, 2, 3], 4.0))))
    out.append(_build("c1", 1102, 5, 3, [], ([0, 1, 2, 3, 4], 1, 13.0, 4.0, fcap([0, 1, 2, 3, 4], 4.0))))
    out.append(_build("c2", 1103, 6, 4, [], ([0, 1, 2, 3, 4, 5], 2, 11.0, 3.0, fcap([0, 1, 2, 3, 4, 5], 3.0))))

    # -- 3..7: TRAP -- a fractional-cycle gadget planted among filler tasks; a task-by-
    # task best-first-fit greedy grabs the wrong side of the cycle and needs the overflow
    # desk for its partner, losing far more than the LP+pipage+repair pipeline does.
    out.append(_build("t3", 1201, 6, 3,
                       [("G1", 0, 1, 0, 1)],
                       ([2, 3, 4, 5], 2, 10.0, 4.0, fcap([2, 3, 4, 5], 4.0))))
    out.append(_build("t4", 1202, 6, 4,
                       [("G2", 0, 1, 0, 1)],
                       ([2, 3, 4, 5], 2, 11.0, 4.0, fcap([2, 3, 4, 5], 4.0))))
    out.append(_build("t5", 1203, 8, 5,
                       [("G3", 0, 1, 0, 1), ("G4", 2, 3, 2, 3)],
                       ([4, 5, 6, 7], 4, 10.0, 4.0, fcap([4, 5, 6, 7], 4.0))))
    out.append(_build("t6", 1204, 6, 3,
                       [("G5", 0, 1, 0, 1)],
                       ([2, 3, 4, 5], 2, 12.0, 4.0, fcap([2, 3, 4, 5], 4.0))))
    out.append(_build("t7", 1205, 8, 5,
                       [("G1", 0, 1, 0, 1), ("G6", 2, 3, 2, 3)],
                       ([4, 5, 6, 7], 4, 10.0, 4.0, fcap([4, 5, 6, 7], 4.0))))

    # -- 8,9: harder / held-out -- more tasks, more agents, multiple gadgets stacked.
    out.append(_build("h8", 1301, 9, 5,
                       [("G2", 0, 1, 0, 1), ("G4", 2, 3, 2, 3)],
                       ([4, 5, 6, 7, 8], 4, 11.0, 4.0, fcap([4, 5, 6, 7, 8], 4.0))))
    out.append(_build("h9", 1302, 10, 7,
                       [("G1", 0, 1, 0, 1), ("G3", 2, 3, 2, 3), ("G5", 4, 5, 4, 5)],
                       ([6, 7, 8, 9], 6, 11.0, 4.0, fcap([6, 7, 8, 9], 4.0))))
    return out


# ----------------------------- answer validation -----------------------------
def _validate_answer(answer, N, M, weight, capacity):
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list) or len(assign) != N:
        return None
    load = [0.0] * M
    out = []
    for i, k in enumerate(assign):
        if isinstance(k, bool) or not isinstance(k, int):
            return None
        if k < 0 or k >= M:
            return None
        out.append(k)
        load[k] += weight[i][k]
    for k in range(M):
        if load[k] > capacity[k] + 1e-6:
            return None
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        N, M = inst["n_tasks"], inst["n_agents"]
        weight, value, capacity = inst["weight"], inst["value"], inst["capacity"]

        base_assign = _naive_baseline(inst)
        obj_base = _value_of(inst, base_assign)
        ref_assign = _internal_reference(inst)
        obj_ref = _value_of(inst, ref_assign)
        denom = 1.3 * (obj_ref - obj_base)
        if denom < 1e-9:
            denom = 1e-9

        public = {"instance_id": inst["name"], "n_tasks": N, "n_agents": M,
                  "capacity": list(capacity),
                  "weight": [list(row) for row in weight],
                  "value": [list(row) for row in value]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            assign = _validate_answer(ans, N, M, weight, capacity)
        except Exception:
            assign = None
        if assign is None:
            vec.append(0.0)
            continue
        try:
            obj_cand = _value_of(inst, assign)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj_cand - obj_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
