# TIER: strong
# INSIGHT (the shifting bottleneck heuristic, not "greedy + more passes"): a machine's
# criticality cannot be judged by its own duration or total load -- it depends on the
# HEADS (earliest possible start, from everything upstream so far) and TAILS (processing
# still required downstream) its operations inherit from the whole routing.  Composed
# moves:
#   (1) ONE-MACHINE-BOTTLENECK-IDENTIFY -- for every still-unsequenced machine, compute
#       heads/tails on the CURRENT partial precedence graph (job edges + machine edges
#       already fixed), then solve that machine's 1|r_i|L_max relaxation with a Schrage
#       heuristic (among released operations, always run the largest-tail one next); its
#       achieved bound max_i(completion_i + tail_i) measures how critical that machine
#       truly is right now.
#   (2) SHIFTING-BOTTLENECK-PRIORITY -- fix the machine with the LARGEST bound first (that
#       is the real bottleneck, not the busiest machine), which changes every other
#       machine's heads/tails, so recompute and repeat until every machine is fixed; then
#       REVISIT each already-fixed machine once more against the completed graph (the
#       ranking can shift again once later machines lock in).
#   (3) CRITICAL-PATH-SWAP -- find the longest chain through the final schedule; whenever
#       two adjacent operations on the SAME machine both sit on that chain, try swapping
#       their order and keep the swap only if it actually shortens the makespan.
# A safety fallback (id-order, always feasible) guards the rare case where the heuristic
# ordering would create a cycle.
import sys, json, heapq
from collections import deque


def build_pred_succ(n_ops, job_ops, dur, fixed_machine_order):
    pred = [[] for _ in range(n_ops)]
    for ids in job_ops:
        for a, b in zip(ids, ids[1:]):
            pred[b].append(a)
    for seq in fixed_machine_order.values():
        if seq is None:
            continue
        for a, b in zip(seq, seq[1:]):
            pred[b].append(a)
    succ = [[] for _ in range(n_ops)]
    for v in range(n_ops):
        for p in pred[v]:
            succ[p].append(v)
    return pred, succ


def heads_tails(n_ops, job_ops, dur, fixed_machine_order):
    pred, succ = build_pred_succ(n_ops, job_ops, dur, fixed_machine_order)
    indeg = [len(pred[i]) for i in range(n_ops)]
    q = deque(i for i in range(n_ops) if indeg[i] == 0)
    order = []
    ind2 = indeg[:]
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            ind2[v] -= 1
            if ind2[v] == 0:
                q.append(v)
    if len(order) != n_ops:
        return None, None
    head = [0.0] * n_ops
    for u in order:
        for v in succ[u]:
            c = head[u] + dur[u]
            if c > head[v]:
                head[v] = c
    tail = [0.0] * n_ops
    for u in reversed(order):
        for v in succ[u]:
            c = tail[v] + dur[v]
            if c > tail[u]:
                tail[u] = c
    return head, tail


def schrage(op_ids, dur, head, tail):
    remaining = sorted(op_ids, key=lambda o: head[o])
    ready = []
    t = head[remaining[0]] if remaining else 0.0
    seq = []
    lmax = 0.0
    ri, n = 0, len(remaining)
    while ri < n or ready:
        while ri < n and head[remaining[ri]] <= t + 1e-9:
            heapq.heappush(ready, (-tail[remaining[ri]], remaining[ri]))
            ri += 1
        if not ready:
            t = head[remaining[ri]]
            continue
        _, o = heapq.heappop(ready)
        start = t
        finish = start + dur[o]
        seq.append(o)
        t = finish
        if finish + tail[o] > lmax:
            lmax = finish + tail[o]
    return seq, lmax


def simulate_full(n_ops, job_ops, dur, machine_order, n_machines):
    fixed = {m: machine_order[m] for m in range(n_machines)}
    pred, succ = build_pred_succ(n_ops, job_ops, dur, fixed)
    indeg = [len(pred[i]) for i in range(n_ops)]
    q = deque(i for i in range(n_ops) if indeg[i] == 0)
    order = []
    ind2 = indeg[:]
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            ind2[v] -= 1
            if ind2[v] == 0:
                q.append(v)
    if len(order) != n_ops:
        return None
    finish = [0.0] * n_ops
    for u in order:
        st = max((finish[p] for p in pred[u]), default=0.0)
        finish[u] = st + dur[u]
    return max(finish)


def main():
    inst = json.load(sys.stdin)
    n_ops = inst["n_ops"]
    n_m = inst["n_machines"]
    job_ops = inst["job_ops"]
    machine_ops = inst["machine_ops"]
    dur = [0.0] * n_ops
    for o in inst["ops"]:
        dur[o["id"]] = o["dur"]

    fixed = {m: None for m in range(n_m)}
    remaining = set(range(n_m))
    cyclic = False
    while remaining:
        head, tail = heads_tails(n_ops, job_ops, dur, fixed)
        if head is None:
            cyclic = True
            break
        best_m, best_bound, best_seq = None, -1.0, None
        for m in list(remaining):
            ops_m = machine_ops[m]
            if not ops_m:
                fixed[m] = []
                remaining.discard(m)
                continue
            seq, lmax = schrage(ops_m, dur, head, tail)
            if lmax > best_bound:
                best_bound, best_m, best_seq = lmax, m, seq
        if best_m is None:
            break
        fixed[best_m] = best_seq
        remaining.discard(best_m)

    if cyclic or any(fixed[m] is None for m in range(n_m)):
        fixed = {m: sorted(machine_ops[m]) for m in range(n_m)}

    machine_order = [fixed[m] for m in range(n_m)]
    ms = simulate_full(n_ops, job_ops, dur, machine_order, n_m)
    if ms is None:
        machine_order = [sorted(machine_ops[m]) for m in range(n_m)]
        ms = simulate_full(n_ops, job_ops, dur, machine_order, n_m)

    # ---- reoptimization: revisit each already-fixed machine against the completed graph
    for _round in range(3):
        improved_any = False
        for m in range(n_m):
            if not machine_ops[m]:
                continue
            other_fixed = {mm: machine_order[mm] for mm in range(n_m) if mm != m}
            head, tail = heads_tails(n_ops, job_ops, dur, other_fixed)
            if head is None:
                continue
            seq, _ = schrage(machine_ops[m], dur, head, tail)
            old_seq = machine_order[m]
            machine_order[m] = seq
            new_ms = simulate_full(n_ops, job_ops, dur, machine_order, n_m)
            if new_ms is not None and new_ms < ms - 1e-9:
                ms = new_ms
                improved_any = True
            else:
                machine_order[m] = old_seq
        if not improved_any:
            break

    # ---- critical-path-swap local search ----
    def critical_path():
        pred, succ = build_pred_succ(n_ops, job_ops, dur,
                                      {m: machine_order[m] for m in range(n_m)})
        indeg = [len(pred[i]) for i in range(n_ops)]
        q = deque(i for i in range(n_ops) if indeg[i] == 0)
        order = []
        ind2 = indeg[:]
        while q:
            u = q.popleft()
            order.append(u)
            for v in succ[u]:
                ind2[v] -= 1
                if ind2[v] == 0:
                    q.append(v)
        if len(order) != n_ops:
            return None
        fin = [0.0] * n_ops
        bp = [None] * n_ops
        for u in order:
            if pred[u]:
                p = max(pred[u], key=lambda x: fin[x])
                fin[u] = fin[p] + dur[u]
                bp[u] = p
            else:
                fin[u] = dur[u]
        end = max(range(n_ops), key=lambda i: fin[i])
        path = [end]
        cur = end
        while bp[cur] is not None:
            cur = bp[cur]
            path.append(cur)
        path.reverse()
        return path

    for _pass in range(6):
        path = critical_path()
        if path is None:
            break
        pos_in_machine = {}
        for m in range(n_m):
            for idx, o in enumerate(machine_order[m]):
                pos_in_machine[o] = (m, idx)
        improved = False
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            mu, iu = pos_in_machine[u]
            mv, iv = pos_in_machine[v]
            if mu == mv and iv == iu + 1:
                seq = machine_order[mu]
                seq[iu], seq[iv] = seq[iv], seq[iu]
                new_ms = simulate_full(n_ops, job_ops, dur, machine_order, n_m)
                if new_ms is not None and new_ms < ms - 1e-9:
                    ms = new_ms
                    improved = True
                    break
                seq[iu], seq[iv] = seq[iv], seq[iu]
        if not improved:
            break

    print(json.dumps({"machine_order": machine_order}))


main()
