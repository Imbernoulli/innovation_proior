# TIER: greedy
# Skill-weighted committee.  Score each member by its Brier skill on the labelled
# validation history and take a weighted average (weak / near-noise members get
# little weight, experts get most).  Beats the equal-weight mean because it stops
# the bad members from dragging the committee down, but it neither recalibrates nor
# jointly stacks the members, so it stays well short of a fitted fuser.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
val_pred = inst["val_pred"]
val_y = inst["val_y"]
test_pred = inst["test_pred"]
nv = len(val_y)

# per-member validation Brier
bs = [0.0] * k
for i in range(nv):
    row = val_pred[i]
    y = val_y[i]
    for j in range(k):
        d = row[j] - y
        bs[j] += d * d
for j in range(k):
    bs[j] /= max(1, nv)

# weight ~ how much better than a coin (0.25 = Brier of constant 0.5); floor at a small eps
w = []
for j in range(k):
    wj = 0.25 - bs[j]
    if wj < 1e-3:
        wj = 1e-3
    w.append(wj)
sw = sum(w)
w = [wj / sw for wj in w]

q = []
for row in test_pred:
    s = 0.0
    for j in range(k):
        s += w[j] * row[j]
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    q.append(s)
print(json.dumps({"forecast": q}))
