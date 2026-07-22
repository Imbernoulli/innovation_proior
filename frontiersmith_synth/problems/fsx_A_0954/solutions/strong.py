# TIER: strong
# Portfolio dispatch: exploit wear-job-affinity deliberately instead of just
# balancing queue length.
#
# For an abrasive job, prefer whichever machine(s) have the LOWEST abrasion
# sensitivity `a` (they dull the least, so they are the cheapest place to
# "sacrifice" for abrasive filler). For a finishing job, prefer whichever
# machine(s) have the HIGHEST polish sensitivity `q` (they self-sharpen the
# most, so routing finishing work there deliberately BUILDS a fast machine
# for future high-weight finishing jobs instead of just balancing today's
# queue). Ties among equally-suited machines (including the fully symmetric
# case where every machine is identical) are broken by shortest queue, so no
# specialization is invented where none is possible.
#
# This is an investment decision, not a load-balancing one: the preferred
# machine is used even if it is not the very least busy one right now --
# UNLESS its queue has drifted far enough ahead of the pack that piling on
# more work would cost more in queueing delay than the future speed
# advantage is worth, in which case a real-time-aware fallback (myopic
# earliest completion, using each machine's ACTUAL current speed) takes
# over as a safety valve.
import sys, json

OVERFLOW = 1.35
SLACK = 1.0

inst = json.load(sys.stdin)
job = inst["job"]
machines = inst["machines"]
typ = job["type"]
p = job["size"]

if typ == "A":
    best_val = min(m["a"] for m in machines)
    tied = [m for m in machines if abs(m["a"] - best_val) < 1e-9]
else:
    best_val = max(m["q"] for m in machines)
    tied = [m for m in machines if abs(m["q"] - best_val) < 1e-9]

pref = min(tied, key=lambda m: (m["free_at"], m["id"]))
min_free = min(m["free_at"] for m in machines)

if pref["free_at"] <= min_free * OVERFLOW + SLACK:
    chosen = pref["id"]
else:
    fallback = min(machines, key=lambda m: (m["free_at"] + p / m["spd"], m["id"]))
    chosen = fallback["id"]

print(json.dumps({"assign": chosen}))
