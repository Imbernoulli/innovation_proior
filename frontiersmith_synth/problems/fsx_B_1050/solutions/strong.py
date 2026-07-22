# TIER: strong
import sys, math

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    ALPHA = float(next(it)); LAMBDA = float(next(it))  # LAMBDA unused: partition
    # choice is driven by the bonus matrix, not by the cost weight; ALPHA is
    # unused directly too -- the scheduling insight (largest-cost-first) is
    # correct for ANY monotonically-decreasing-then-increasing weight shape.
    pts = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    bonus = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            bonus[i][j] = int(next(it))

    # ---- Step 1: pick the FINAL K-cluster partition using the bonus matrix
    # alone (correlation-clustering style greedy on the average inter-group
    # bonus). This is the core insight: judge every merge by the K-cluster
    # cut it is steering toward, not by nearest available pair.
    groups = {i: [i] for i in range(1, N + 1)}
    active_g = list(range(1, N + 1))

    def group_affinity(g1, g2):
        return sum(bonus[x - 1][y - 1] for x in groups[g1] for y in groups[g2])

    while len(active_g) > K:
        best = None
        for ii in range(len(active_g)):
            for jj in range(ii + 1, len(active_g)):
                g1, g2 = active_g[ii], active_g[jj]
                sc = group_affinity(g1, g2) / (len(groups[g1]) * len(groups[g2]))
                if best is None or sc > best[0]:
                    best = (sc, g1, g2)
        _, g1, g2 = best
        groups[g1] = groups[g1] + groups[g2]
        del groups[g2]
        active_g.remove(g2)

    target_of = {}
    for gi, g in enumerate(active_g):
        for p in groups[g]:
            target_of[p] = gi

    # ---- Step 2: within each target cluster, build a local nearest-centroid
    # merge tree (cheapest way to physically assemble that target cluster).
    info = {}
    for i in range(1, N + 1):
        x, y = pts[i - 1]
        info[i] = {"cnt": 1, "sx": float(x), "sy": float(y)}
    next_id = N + 1
    ready = []  # list of dict(id, a, b, cost)
    for gi in range(len(active_g)):
        members = [p for p in range(1, N + 1) if target_of[p] == gi]
        local = list(members)
        while len(local) > 1:
            best = None
            for ii in range(len(local)):
                for jj in range(ii + 1, len(local)):
                    a, b = local[ii], local[jj]
                    ia, ib = info[a], info[b]
                    cxA, cyA = ia["sx"] / ia["cnt"], ia["sy"] / ia["cnt"]
                    cxB, cyB = ib["sx"] / ib["cnt"], ib["sy"] / ib["cnt"]
                    d = math.hypot(cxA - cxB, cyA - cyB)
                    raw = (ia["cnt"] * ib["cnt"] / (ia["cnt"] + ib["cnt"])) * d
                    if best is None or raw < best[0]:
                        best = (raw, a, b)
            raw, a, b = best
            new_id = next_id
            next_id += 1
            ia, ib = info[a], info[b]
            info[new_id] = {"cnt": ia["cnt"] + ib["cnt"], "sx": ia["sx"] + ib["sx"], "sy": ia["sy"] + ib["sy"]}
            ready.append({"id": new_id, "a": a, "b": b, "cost": raw})
            local.remove(a); local.remove(b); local.append(new_id)

    # ---- Step 3: globally SCHEDULE all N-K merges (interleaving across
    # target clusters) by descending raw cost among currently-ready merges.
    # By the rearrangement inequality, since the horizon weight strictly
    # decreases as "remaining merges" grows, pairing the largest costs with
    # the earliest (lowest-weight) slots minimizes the total weighted cost
    # for this fixed set of physical merges -- exploiting the lookahead
    # mechanism instead of ignoring it.
    materialized = set(range(1, N + 1))
    pending = list(ready)
    internal_order = []
    while pending:
        avail = [m for m in pending if m["a"] in materialized and m["b"] in materialized]
        avail.sort(key=lambda m: -m["cost"])
        chosen = avail[0]
        internal_order.append((chosen["a"], chosen["b"], chosen["id"]))
        materialized.add(chosen["id"])
        pending.remove(chosen)

    remap = {}
    scheduled = []
    for pos, (a, b, iid) in enumerate(internal_order, start=1):
        fa = remap.get(a, a)
        fb = remap.get(b, b)
        scheduled.append((fa, fb))
        remap[iid] = N + pos

    out = [str(len(scheduled))]
    for a, b in scheduled:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
