# TIER: greedy
# The obvious recipe (the trap): every job carries positive value, so "admit anything
# with positive value" admits EVERYTHING.  It never looks at congestion or at the
# public future arrivals.  On an uncongested stream this is fine and banks all the
# value; but when a burst of small low-value jobs arrives just before a high-value job
# with a tight deadline, the burst busies the single FIFO server and the premium job
# starts late -- so it is BOTH forfeited (no value) and charged the ramped SLA penalty.
# The gate inflicts its own congestion by refusing to refuse.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"admit": [1] * N}))
