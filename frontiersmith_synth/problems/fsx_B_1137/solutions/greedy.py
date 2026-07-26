# TIER: greedy
# The classic first-instinct scheduling heuristic: shortest-processing-time (SPT) sort.
# It reasons about each job's OWN attribute (proc_time) in isolation and completely
# ignores the sequence-dependent setup/warm-up state and tool-fatigue limit that every
# station accumulates from the shared global order. On instances where proc_time isn't
# a clean proxy for job type (or where a station's fatigue limit conflicts with the long
# same-type runs SPT happens to produce), this commits the line to expensive cold/fatigue
# states it never modeled.
import sys, json

inst = json.load(sys.stdin)
jobs = inst["jobs"]
by_id = {j["id"]: j for j in jobs}
order = sorted(by_id.keys(), key=lambda i: (by_id[i]["proc_time"], i))
print(json.dumps({"order": order}))
