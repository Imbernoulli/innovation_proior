# TIER: greedy
# Shortest-queue dispatch: send this job to whichever machine currently
# becomes free soonest (smallest free_at), ties broken by lowest id. This is
# the obvious "average strong coder" recipe -- it reacts to CURRENT load but
# never looks at a machine's abrasion/polish sensitivity or the job's type,
# so it never deliberately shapes any machine's future speed. It just
# load-balances queue length, which spreads wear (good and bad) evenly
# across every machine instead of concentrating it.
import sys, json

inst = json.load(sys.stdin)
machines = inst["machines"]
best = min(machines, key=lambda m: (m["free_at"], m["id"]))
print(json.dumps({"assign": best["id"]}))
