# TIER: greedy
# MEMORIZATION baseline: 1-nearest-neighbour in raw tag-COUNT space.  For each test
# log, build its per-tag count vector and copy the label of the training log with the
# closest (Euclidean) count vector.  On IN-DISTRIBUTION (short) logs the surface
# statistics resemble training, so this beats the constant predictor -- but on the
# LONG out-of-distribution logs every count vector is far larger than anything seen in
# training, so the nearest neighbour collapses onto whichever training log had the
# biggest counts and the label is usually wrong.  Classic memorization that fails to
# extrapolate with length.
import sys, json

inst = json.load(sys.stdin)
A = inst["A"]


def counts(seq):
    c = [0] * A
    for s in seq:
        c[s] += 1
    return c


train_c = [counts(ex["seq"]) for ex in inst["train"]]
train_y = [ex["label"] for ex in inst["train"]]

labels = []
for seq in inst["test"]:
    tc = counts(seq)
    best_i = 0
    best_d = None
    for i, cc in enumerate(train_c):
        d = 0
        for a in range(A):
            diff = tc[a] - cc[a]
            d += diff * diff
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    labels.append(train_y[best_i])

print(json.dumps({"labels": labels}))
