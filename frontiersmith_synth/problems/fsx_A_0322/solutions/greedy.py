# TIER: greedy
# Weighted-shortest-processing-time (WSPT / Smith's ratio) dispatch: give priority
# to cuts with the largest weight-per-hump-time w/p.  This clears heavy, quick cuts
# first and is a classic single-machine rule that sharply cuts weighted lateness
# versus FCFS -- but it ignores the actual cut-off times and release staggering, so
# it leaves clear headroom above.
import sys, json

inst = json.load(sys.stdin)
cuts = inst["cuts"]
order = sorted(range(len(cuts)),
               key=lambda i: cuts[i]["w"] / cuts[i]["p"], reverse=True)
print(json.dumps({"order": order}))
