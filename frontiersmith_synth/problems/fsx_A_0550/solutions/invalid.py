# TIER: invalid
# Overcommit: dispatch more simultaneous builds than there are crews.  All breaks share
# dispatch time 0, so every build interval [0, b) overlaps -> overlap = crews + 5 > crews,
# violating the lock-in capacity constraint.  The evaluator rejects it and scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; C = inst["crews"]
# crews+5 distinct cells, all dispatched at t=0 -> capacity violated.
breaks = []
r = c = 0
while len(breaks) < C + 5:
    breaks.append([r, c, 0])
    c += 1
    if c >= n:
        c = 0; r += 1
print(json.dumps({"breaks": breaks}))
