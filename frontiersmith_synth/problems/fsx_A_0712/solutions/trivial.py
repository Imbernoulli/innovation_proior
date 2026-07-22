# TIER: trivial
# Weak reference: immediate, length-2-cycles ONLY, first-found, no hoarding,
# no length-3 cycles ever considered.  This exactly mirrors the evaluator's
# weak baseline policy, so it scores about 0.1.
import sys, json

inst = json.load(sys.stdin)
R = inst["n_rounds"]
arrivals = inst["arrivals"]

by_id = {}
for rnd in arrivals:
    for a in rnd:
        by_id[a["id"]] = a

matched = set()
present = set()
out_rounds = [[] for _ in range(R)]

for r in range(R):
    for a in arrivals[r]:
        present.add(a["id"])
    present = {i for i in present
               if by_id[i]["round"] + by_id[i]["patience"] - 1 >= r and i not in matched}
    pool = sorted(present)
    by_pair = {}
    for i in pool:
        a = by_id[i]
        by_pair.setdefault((a["have"], a["want"]), []).append(i)
    cycles = []
    for i in pool:
        if i in matched:
            continue
        a = by_id[i]
        rec = (a["want"], a["have"])
        lst = by_pair.get(rec)
        if lst:
            j = None
            for cand in lst:
                if cand not in matched and cand != i:
                    j = cand
                    break
            if j is not None:
                matched.add(i)
                matched.add(j)
                cycles.append([i, j])
    out_rounds[r] = cycles

print(json.dumps({"rounds": [{"cycles": c} for c in out_rounds]}))
