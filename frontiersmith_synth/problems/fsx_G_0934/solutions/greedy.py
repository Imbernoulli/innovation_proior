# TIER: greedy
# The obvious "textbook" approach: build the literal PREFIX-TREE ACCEPTOR (PTA) --
# one state per distinct prefix seen in the training sample, transitions exactly
# following the tree, and a single REJECT SINK catching every continuation that was
# never observed (so the automaton is complete). This reproduces every training
# label perfectly, but it has (roughly) as many states as there are distinct
# training prefixes, and any held-out trace that runs off the memorized prefixes
# falls straight into the reject sink -- it never generalizes past what it literally
# saw, so it collapses on devices tested with longer held-out traces.
import sys, json

inst = json.load(sys.stdin)
train = inst["train"]

prefixes = {""}
for t in train:
    s = t["s"]
    for i in range(1, len(s) + 1):
        prefixes.add(s[:i])

order = sorted(prefixes, key=lambda p: (len(p), p))
idx = {p: i for i, p in enumerate(order)}
n = len(order)

label = {}
for t in train:
    label[t["s"]] = t["label"]

SINK = n
delta = [[None, None] for _ in range(n + 1)]
for p in prefixes:
    i = idx[p]
    for c in ("0", "1"):
        q = p + c
        delta[i][int(c)] = idx[q] if q in prefixes else SINK
delta[SINK] = [SINK, SINK]

accept = [idx[s] for s, y in label.items() if y == 1]
print(json.dumps({"delta": delta, "start": idx[""], "accept": accept}))
