# TIER: strong
# RULE RECOVERY: reconstruct the hidden per-tag net-flow weights, then apply the exact
# classification rule to any length.  Each training log with count vector c and label y
# constrains the integer weight vector w (total = c . w) by:
#     y == 0  ->  total <= -T-1
#     y == 1  ->  -T <= total <= T
#     y == 2  ->  total >=  T+1
# With A <= 5 tags and weights in {-3..3} the whole hypothesis space is tiny (7^A), so
# we enumerate it, keep every weight vector consistent with ALL training logs, and pick
# the SIMPLEST one (minimum L1 norm, ties broken lexicographically) -- an Occam bias
# toward small flows.  Because this reasons about the underlying rule rather than the
# surface counts, it extrapolates to the long OOD logs.  It is not perfect: the tolerance
# band leaves integer slack, so on some instances the simplest consistent rule disagrees
# with the true weights on a few long logs -> below-1.0 score (headroom).
import sys, json
from itertools import product

inst = json.load(sys.stdin)
A = inst["A"]
T = inst["T"]


def counts(seq):
    c = [0] * A
    for s in seq:
        c[s] += 1
    return c


train_c = [counts(ex["seq"]) for ex in inst["train"]]
train_y = [ex["label"] for ex in inst["train"]]


def consistent(w):
    for c, y in zip(train_c, train_y):
        tot = 0
        for a in range(A):
            tot += c[a] * w[a]
        if y == 0:
            if tot > -T - 1:
                return False
        elif y == 2:
            if tot < T + 1:
                return False
        else:  # y == 1
            if tot < -T or tot > T:
                return False
    return True


best_w = None
best_key = None
for w in product(range(-3, 4), repeat=A):
    if consistent(w):
        key = (sum(abs(x) for x in w), w)  # min L1, then lexicographic
        if best_key is None or key < best_key:
            best_key = key
            best_w = w

if best_w is None:
    # No consistent rule found: fall back to majority-of-train.
    cnt = {0: 0, 1: 0, 2: 0}
    for y in train_y:
        cnt[y] += 1
    maj = 0
    for lab in (0, 1, 2):
        if cnt[lab] > cnt[maj]:
            maj = lab
    print(json.dumps({"labels": [maj] * len(inst["test"])}))
    sys.exit(0)


def classify(seq):
    tot = 0
    for s in seq:
        tot += best_w[s]
    if tot < -T:
        return 0
    if tot > T:
        return 2
    return 1


labels = [classify(seq) for seq in inst["test"]]
print(json.dumps({"labels": labels}))
