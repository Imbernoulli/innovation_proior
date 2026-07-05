# TIER: strong
# Tracks the LIFO stack correctly, then learns the full hidden local law keyed
# on the (top, under) pair -- exactly the law's signature -- with stupid-backoff
# (pair -> top-only -> global mode) for pairs unseen in the limited, skewed
# training data.  Because the law is local/length-free it transfers to the
# longer OOD manifests; the ceiling comes from (top,under) coverage on the
# deeper OOD stacks, so smarter smoothing/backoff can still improve -- genuinely
# open-ended and never perfect.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
QUAY = inst["quay"]

P = {}          # (top, under) -> Counter(code)
U = {}          # top -> Counter(code)
glob = Counter()

def replay(moves, codes):
    stack = []
    ci = 0
    for mv in moves:
        if mv == -1:
            top = stack.pop()
            under = stack[-1] if stack else QUAY
            code = codes[ci]; ci += 1
            P.setdefault((top, under), Counter())[code] += 1
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
            under = stack[-1] if stack else QUAY
            if (top, under) in P:
                out.append(P[(top, under)].most_common(1)[0][0])
            elif top in U:
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
