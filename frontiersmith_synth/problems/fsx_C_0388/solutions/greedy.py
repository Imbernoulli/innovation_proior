# TIER: greedy
# Overfit the single most ID-predictive feature: `length` (index 3). Negatives run longer than
# positives IN-DISTRIBUTION, so a length threshold nails the ID bucket -- and collapses to chance
# on the LENGTH-OOD buckets where every log is long. Classic memorize-the-spurious-cue failure.
import sys, json
inst = json.load(sys.stdin)
m = inst["m"]
LEN = 3
train = inst["train"]
pos = [row[LEN] for row in train if row[-1] == 1]
neg = [row[LEN] for row in train if row[-1] == 0]
mp = sum(pos) / len(pos)
mn = sum(neg) / len(neg)
thr = 0.5 * (mp + mn)
w = [0.0] * m
if mn >= mp:                 # negatives longer -> predict VALID when length < thr
    w[LEN] = -1.0
    b = thr
else:                        # (reverse orientation) predict VALID when length > thr
    w[LEN] = 1.0
    b = -thr
print(json.dumps({"w": w, "b": b}))
