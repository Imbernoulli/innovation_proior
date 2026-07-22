# TIER: trivial
# Ignore position entirely: one gene that never fires, so every position (and
# every knockout, since there is nothing left to knock out) decodes to
# whichever single type is the plan's most common ("always guess the mode").
# A network that reads no positional structure at all.
import sys, json

inst = json.load(sys.stdin)
target = inst["target"]
K = inst["K"]
T = inst["T"]

counts = [0] * K
for t in target:
    counts[t] += 1
mode = max(range(K), key=lambda k: counts[k])

G = 1
Win = [[0] * T]
W = [[0]]
bias = [-1]           # raw = -1 <= 0 always -> the gene never fires
decode = [mode, mode]

print(json.dumps({"G": G, "Win": Win, "W": W, "bias": bias, "decode": decode}))
