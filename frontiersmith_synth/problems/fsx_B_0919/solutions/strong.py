# TIER: strong
# Exploits BOTH planted mechanisms via a genuine insight: instead of a
# crude private per-node heat proxy, actually RUN the same
# diffusion+hysteresis physics the evaluator uses (every coefficient is
# given in the input) to score every candidate (node, start-time) pair
# for the next job, over a delay window, against the jobs already
# committed -- and score it by the TRUE MARGINAL VALUE: total physics
# contribution across ALL committed jobs WITH this placement minus
# WITHOUT it. That marginal-value framing prices in the externality a
# placement inflicts on already-committed neighbor jobs (via diffusion),
# not just the new job's own outcome -- so the policy will refuse a
# nominally "coolest" node if committing there would tip a shared
# neighborhood into a hysteresis lock. It will also happily DELAY a job
# (preemptively leave a node idle) rather than always start ASAP into a
# forming hotspot. This is a decomposition/marginal-value exchange
# argument, not "greedy plus more iterations".
import sys, json

def simulate_contrib(N, T, decay, alpha, HI, LO, trate, neigh, occ, heat_rate_of):
    H = [0.0] * N
    throttled = [False] * N
    contrib = {}
    for t in range(T):
        m = [trate if throttled[i] else 1.0 for i in range(N)]
        gen = [0.0] * N
        for i in range(N):
            jid = occ[i][t]
            if jid is not None:
                gen[i] = heat_rate_of[jid] * m[i]
                contrib[jid] = contrib.get(jid, 0.0) + m[i]
        newH = [0.0] * N
        for i in range(N):
            diff = 0.0
            for k in neigh[i]:
                diff += H[k] - H[i]
            newH[i] = H[i] + gen[i] - decay * H[i] + alpha * diff
        H = newH
        for i in range(N):
            if throttled[i]:
                if H[i] <= LO:
                    throttled[i] = False
            else:
                if H[i] > HI:
                    throttled[i] = True
    return contrib


def main():
    inst = json.load(sys.stdin)
    N, T = inst["N"], inst["T"]
    rows, cols = inst["grid_rows"], inst["grid_cols"]
    decay, alpha, HI, LO, trate = inst["decay"], inst["alpha"], inst["HI"], inst["LO"], inst["throttle_rate"]
    jobs = list(inst["jobs"])
    heat_rate_of = {j["id"]: j["heat_rate"] for j in jobs}

    neigh = [[] for _ in range(N)]
    for i in range(N):
        r, c = divmod(i, cols)
        if r > 0: neigh[i].append(i - cols)
        if r < rows - 1: neigh[i].append(i + cols)
        if c > 0: neigh[i].append(i - 1)
        if c < cols - 1: neigh[i].append(i + 1)

    order = sorted(jobs, key=lambda j: (j["arrival"], j["id"]))

    occ = [[None] * T for _ in range(N)]
    node_free = [0] * N
    committed_node = {}
    committed_start = {}

    DELAY_WINDOW = 18
    DELAY_STEP = 2
    LAMBDA = 0.005  # tiny time-preference: break near-ties toward earlier start

    for j in order:
        jid, arrival, demand = j["id"], j["arrival"], j["demand"]

        base_contrib = simulate_contrib(N, T, decay, alpha, HI, LO, trate, neigh, occ, heat_rate_of)
        base_total = sum(base_contrib.values())

        best = None  # (marginal_value, node, start)
        for node in range(N):
            base_start = max(arrival, node_free[node])
            for delay in range(0, DELAY_WINDOW + 1, DELAY_STEP):
                start = base_start + delay
                if start + demand > T:
                    continue
                ok = True
                for t in range(start, start + demand):
                    if occ[node][t] is not None:
                        ok = False
                        break
                if not ok:
                    continue
                for t in range(start, start + demand):
                    occ[node][t] = jid
                new_contrib = simulate_contrib(N, T, decay, alpha, HI, LO, trate, neigh, occ, heat_rate_of)
                new_total = sum(new_contrib.values())
                for t in range(start, start + demand):
                    occ[node][t] = None
                marginal = (new_total - base_total) - LAMBDA * delay
                if best is None or marginal > best[0]:
                    best = (marginal, node, start)

        if best is None or best[0] <= 1e-9:
            continue  # not worth scheduling anywhere within the window
        _, node, start = best
        for t in range(start, start + demand):
            occ[node][t] = jid
        node_free[node] = max(node_free[node], start + demand)
        committed_node[jid] = node
        committed_start[jid] = start

    sched = []
    for j in jobs:
        jid = j["id"]
        if jid in committed_node:
            sched.append({"id": jid, "node": committed_node[jid], "start": committed_start[jid]})
        else:
            sched.append({"id": jid, "node": -1, "start": 0})

    print(json.dumps({"schedule": sched}))

main()
