# TIER: greedy
# Ratio-greedy budgeted coverage: repeatedly build the affordable site that adds
# the most NEW watched weight per unit cost, until nothing else fits the budget.
# This actually looks at coverage (unlike the cost-ascending fill) and reuses the
# marginal-gain idea behind the 1-1/e greedy, but it never revises a choice, so
# early ratio-optimal picks that overlap later value leave weight uncollected.
import sys, json

inst = json.load(sys.stdin)
N, M, B = inst["N"], inst["M"], inst["B"]
W, tx, ty, tr, tc = inst["weight"], inst["tx"], inst["ty"], inst["tr"], inst["tc"]


def rect(j):
    r = tr[j]
    r0 = max(0, ty[j] - r); r1 = min(N - 1, ty[j] + r)
    c0 = max(0, tx[j] - r); c1 = min(N - 1, tx[j] + r)
    return r0, r1, c0, c1


seen = [[False] * N for _ in range(N)]
spent = 0
build = []
avail = set(range(M))

while True:
    best_j = -1
    best_ratio = 0.0
    best_gain = 0
    for j in avail:
        if spent + tc[j] > B:
            continue
        r0, r1, c0, c1 = rect(j)
        gain = 0
        for r in range(r0, r1 + 1):
            sr = seen[r]; wr = W[r]
            for c in range(c0, c1 + 1):
                if not sr[c]:
                    gain += wr[c]
        if gain <= 0:
            continue
        ratio = gain / tc[j]
        if ratio > best_ratio + 1e-12:
            best_ratio = ratio; best_j = j; best_gain = gain
    if best_j < 0:
        break
    r0, r1, c0, c1 = rect(best_j)
    for r in range(r0, r1 + 1):
        sr = seen[r]
        for c in range(c0, c1 + 1):
            sr[c] = True
    spent += tc[best_j]
    build.append(best_j)
    avail.discard(best_j)

print(json.dumps({"build": build}))
