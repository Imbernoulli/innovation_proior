# TIER: greedy
# Tracks the LIFO stack correctly (so the removed container's bay-type `top` is
# right at any length), but learns a coarse map keyed on `top` ONLY, ignoring
# the container underneath.  Captures the dominant mass but misses the true
# (top,under) dependence -> beats trivial, below strong.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
QUAY = inst["quay"]

U = {}          # top -> Counter(code)
glob = Counter()

def replay(moves, codes):
    stack = []
    ci = 0
    for mv in moves:
        if mv == -1:
            top = stack.pop()
            code = codes[ci]; ci += 1
            U.setdefault(top, Counter())[code] += 1
            glob[code] += 1
        else:
            stack.append(mv)

for tr in inst["train"]:
    replay(tr["moves"], tr["codes"])

gm = glob.most_common(1)[0][0] if glob else 0

def predict(moves):
    stack = []
    out = []
    for mv in moves:
        if mv == -1:
            top = stack.pop()
            if top in U:
                out.append(U[top].most_common(1)[0][0])
            else:
                out.append(gm)
        else:
            stack.append(mv)
    return out

q = inst["queries"]
ans = {"predictions": {
    "id": [predict(mv) for mv in q["id"]],
    "ood": [predict(mv) for mv in q["ood"]],
}}
print(json.dumps(ans))
