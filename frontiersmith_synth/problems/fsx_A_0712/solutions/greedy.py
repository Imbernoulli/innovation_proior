# TIER: greedy
# Textbook per-round matcher: form every length-2 cycle available the instant
# it appears, then mop up remaining length-3 cycles -- both computed to
# completion each round, but with NO memory of urgency and NO hoarding.
# It clears ordinary rounds easily (finds every currently-possible cycle),
# but it burns a scarce Moon-and-Stars holder on the first reciprocal
# partner it meets, foreclosing a bigger cycle one round later that would
# have used the SAME holder to rescue more gardeners at once.
import sys, json

inst = json.load(sys.stdin)
T = inst["n_types"]
R = inst["n_rounds"]
arrivals = inst["arrivals"]

by_id = {}
for rnd in arrivals:
    for a in rnd:
        by_id[a["id"]] = a

matched = set()
present = set()
out_rounds = [[] for _ in range(R)]


def fire_2cycles(pool):
    by_pair = {}
    for i in pool:
        if i in matched:
            continue
        a = by_id[i]
        by_pair.setdefault((a["have"], a["want"]), []).append(i)
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


def fire_3cycles(pool):
    by_pair = {}
    for i in pool:
        if i in matched:
            continue
        a = by_id[i]
        by_pair.setdefault((a["have"], a["want"]), []).append(i)
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


for r in range(R):
    for a in arrivals[r]:
        present.add(a["id"])
    present = {i for i in present
               if by_id[i]["round"] + by_id[i]["patience"] - 1 >= r and i not in matched}
    pool = sorted(present)
    c2 = fire_2cycles(pool)
    pool2 = [i for i in pool if i not in matched]
    c3 = fire_3cycles(pool2)
    out_rounds[r] = c2 + c3

print(json.dumps({"rounds": [{"cycles": c} for c in out_rounds]}))
