# TIER: greedy
# Suffix-window classifier.  Build a DFA whose state remembers only the last k modules
# of the trellis; label each window-state by the majority training label of trellises
# ending in that window.  This captures LOCAL suffix patterns, so it generalizes a bit
# to taller trellises -- better than the majority baseline -- but it cannot represent
# the global finite-state structure of the true rule, so it stays well below a real
# grammar-induction learner.
import sys, json
from itertools import product

inst = json.load(sys.stdin)
D = inst["n_types"]
M = inst["max_states"]
train = inst["train"]

# largest window k whose state count stays within budget
k = 1
while True:
    tot = sum(D ** i for i in range(k + 2))
    if k + 1 <= 8 and tot <= min(M, 48):
        k += 1
    else:
        break

# states = all module tuples of length 0..k (0 = empty = start)
states = []
for L in range(k + 1):
    for combo in product(range(D), repeat=L):
        states.append(combo)
idx = {s: i for i, s in enumerate(states)}
K = len(states)


def nxt(suf, c):
    t = suf + (c,)
    if len(t) > k:
        t = t[-k:]
    return t


trans = [[idx[nxt(s, c)] for c in range(D)] for s in states]


def final_state(seq):
    suf = ()
    for c in seq:
        suf = nxt(suf, c)
    return idx[suf]


counts = {}
for seq, y in train:
    fs = final_state(seq)
    cc = counts.get(fs)
    if cc is None:
        cc = [0, 0]
        counts[fs] = cc
    cc[y] += 1

ones = sum(1 for _, y in train if y == 1)
gmaj = 1 if ones * 2 >= len(train) else 0

accept = []
for i in range(K):
    cc = counts.get(i)
    if cc is None:
        accept.append(gmaj)
    else:
        accept.append(1 if cc[1] >= cc[0] else 0)

print(json.dumps({"start": idx[()], "accept": accept, "trans": trans}))
