# TIER: strong
"""
Insight: relax the mandatory-total-1-per-task, capacity-per-desk assignment LP, round the
fractional vertex along its cycles two variables (desks) at a time -- keeping whichever of
the cycle's two alternating colorings earns more value -- and finally patch up any residual
capacity overshoot with local value-improving swaps toward the overflow desk.
"""
import sys, json, os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TOL = 1e-6


def lp_relax(N, M, weight, capacity, value):
    from scipy.optimize import linprog
    nv = N * M
    c = [0.0] * nv
    for i in range(N):
        for k in range(M):
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


def pipage_round(N, M, weight, value, capacity, x):
    assign = [None] * N
    frac = {}
    load = [0.0] * M
    for i in range(N):
        fk = [k for k in range(M) if TOL < x[i][k] < 1.0 - TOL]
        if fk:
            frac[i] = set(fk)
        else:
            k1 = max(range(M), key=lambda k: x[i][k])
            assign[i] = k1
            load[k1] += weight[i][k1]

    for i in list(frac.keys()):
        while len(frac[i]) > 2:
            keep = set(sorted(frac[i], key=lambda k: -value[i][k])[:2])
            frac[i] = keep

    agent_tasks = {}
    for i, ks in frac.items():
        for k in ks:
            agent_tasks.setdefault(k, set()).add(i)

    steps = 0
    while frac and steps <= N + 5:
        steps += 1
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

    for i in list(frac.keys()):
        assign[i] = max(frac[i], key=lambda k: value[i][k])
    for i in range(N):
        if assign[i] is None:
            assign[i] = M - 1
    return assign


def swap_repair(N, M, weight, capacity, value, assign, max_passes=200):
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


def main():
    inst = json.load(sys.stdin)
    N, M = inst["n_tasks"], inst["n_agents"]
    weight, value, capacity = inst["weight"], inst["value"], inst["capacity"]
    x = lp_relax(N, M, weight, capacity, value)
    assign = pipage_round(N, M, weight, value, capacity, x)
    assign = swap_repair(N, M, weight, capacity, value, assign)
    print(json.dumps({"assign": [int(a) for a in assign]}))


if __name__ == "__main__":
    main()
