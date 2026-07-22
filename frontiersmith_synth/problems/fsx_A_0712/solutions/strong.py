# TIER: strong
# Full-lookahead, type-flow-aware scheduler.
#   (1) Force-resolve any gardener at their LAST valid round with the
#       biggest cycle available now -- otherwise they're lost for good.
#   (2) Fire every available length-3 cycle before any length-2 cycle, so a
#       chain is never pre-empted by an easy pair.
#   (3) Count have/want per type across the WHOLE known season (the input
#       already contains every round) to find which types are chronically
#       oversubscribed (demand > supply).  Never spend a non-forced holder
#       of such a type on a plain length-2 cycle -- bank it until either a
#       bigger cycle opens up or its own deadline forces the issue.
# This type-flow-conservation insight -- computing exactly which types are
# worth hoarding -- strictly dominates matching everything the instant it
# becomes possible.
import sys, json

inst = json.load(sys.stdin)
T = inst["n_types"]
R = inst["n_rounds"]
arrivals = inst["arrivals"]

by_id = {}
for rnd in arrivals:
    for a in rnd:
        by_id[a["id"]] = a

supply = [0] * T
demand = [0] * T
for a in by_id.values():
    supply[a["have"]] += 1
    demand[a["want"]] += 1
scarce = {t for t in range(T) if supply[t] < demand[t]}

matched = set()
present = set()
out_rounds = [[] for _ in range(R)]


def by_pair_index(pool):
    d = {}
    for i in pool:
        if i in matched:
            continue
        a = by_id[i]
        d.setdefault((a["have"], a["want"]), []).append(i)
    return d


def find_cycle_for(agent_id, pool_set):
    a = by_id[agent_id]
    ha, wa = a["have"], a["want"]
    others = [i for i in pool_set if i != agent_id and i not in matched]
    by_have = {}
    for i in others:
        by_have.setdefault(by_id[i]["have"], []).append(i)
    # 3-cycle: agent_id <- i1 <- i2 <- agent_id
    for i1 in sorted(by_have.get(wa, [])):
        w1 = by_id[i1]["want"]
        for i2 in sorted(by_have.get(w1, [])):
            if i2 == i1:
                continue
            if by_id[i2]["want"] == ha:
                return [agent_id, i1, i2]
    # 2-cycle: direct reciprocal partner
    for i in sorted(by_have.get(wa, [])):
        if by_id[i]["want"] == ha:
            return [agent_id, i]
    return None


def fire_3cycles(pool):
    by_pair = by_pair_index(pool)
    cycles = []
    for t0 in range(T):
        for t1 in range(T):
            if t1 == t0:
                continue
            for t2 in range(T):
                if t2 == t0 or t2 == t1:
                    continue
                while True:
                    l0 = [x for x in by_pair.get((t0, t1), []) if x not in matched]
                    l1 = [x for x in by_pair.get((t1, t2), []) if x not in matched]
                    l2 = [x for x in by_pair.get((t2, t0), []) if x not in matched]
                    if not l0 or not l1 or not l2:
                        break
                    a0, a1, a2 = l0[0], l1[0], l2[0]
                    if len({a0, a1, a2}) < 3:
                        break
                    matched.add(a0)
                    matched.add(a1)
                    matched.add(a2)
                    cycles.append([a0, a1, a2])
    return cycles


def fire_2cycles(pool):
    by_pair = by_pair_index(pool)
    cycles = []
    for (h, w) in sorted(by_pair.keys()):
        if h >= w:
            continue
        la = [x for x in by_pair.get((h, w), []) if x not in matched]
        lb = [x for x in by_pair.get((w, h), []) if x not in matched]
        ia = ib = 0
        while ia < len(la) and ib < len(lb):
            i = la[ia]
            j = lb[ib]
            if i in matched:
                ia += 1
                continue
            if j in matched:
                ib += 1
                continue
            matched.add(i)
            matched.add(j)
            cycles.append([i, j])
            ia += 1
            ib += 1
    return cycles


for r in range(R):
    for a in arrivals[r]:
        present.add(a["id"])
    present = {i for i in present
               if by_id[i]["round"] + by_id[i]["patience"] - 1 >= r and i not in matched}

    cycles_this_round = []

    # (a) forced: gardeners whose last valid round is today
    forced = sorted(i for i in present
                     if by_id[i]["round"] + by_id[i]["patience"] - 1 == r and i not in matched)
    for i in forced:
        if i in matched:
            continue
        cyc = find_cycle_for(i, present)
        if cyc is not None and all(x not in matched for x in cyc):
            for x in cyc:
                matched.add(x)
            cycles_this_round.append(cyc)

    # (b) every available length-3 cycle, before any length-2 cycle
    pool = [i for i in present if i not in matched]
    cycles_this_round += fire_3cycles(pool)

    # (c) length-2 cycles, holding non-forced scarce-`have` gardeners back
    pool = [i for i in present if i not in matched]
    held = {i for i in pool if by_id[i]["have"] in scarce
            and by_id[i]["round"] + by_id[i]["patience"] - 1 != r}
    pool_c = [i for i in pool if i not in held]
    cycles_this_round += fire_2cycles(pool_c)

    # (d) mop up any length-3 cycle exposed by step (c)
    pool = [i for i in present if i not in matched]
    cycles_this_round += fire_3cycles(pool)

    out_rounds[r] = cycles_this_round

print(json.dumps({"rounds": [{"cycles": c} for c in out_rounds]}))
