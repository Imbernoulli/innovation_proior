# TIER: greedy
# The obvious recipe: earliest-deadline-first (EDF). Sort all jobs by deadline and
# submit them in that order, relying on the evaluator's admission control to skip
# anything infeasible. This is the textbook single-machine deadline-scheduling
# heuristic -- but it is completely family-blind: it never considers that switching
# families costs a setup, so on a trap instance where family assignment interleaves
# with deadline rank it pays a setup on almost every job and thrashes.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
d = inst["d"]

order = sorted(range(n), key=lambda i: (d[i], i))
print(json.dumps({"order": order}))
