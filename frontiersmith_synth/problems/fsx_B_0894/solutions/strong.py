# TIER: strong
import sys, json, math


def waterfill(g, neff, P):
    """Water-fill against an arbitrary (possibly interference-inflated) noise
    vector `neff` instead of the raw noise floor."""
    N = len(g)
    lo, hi = 0.0, max(neff[i] / g[i] for i in range(N)) + P + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = sum(max(0.0, mid - neff[i] / g[i]) for i in range(N))
        if s > P:
            hi = mid
        else:
            lo = mid
    mu = (lo + hi) / 2.0
    p = [max(0.0, mu - neff[i] / g[i]) for i in range(N)]
    tot = sum(p)
    if tot > P and tot > 1e-12:
        scale = P / tot
        p = [x * scale for x in p]
    return p


def rate_given(g, n, A, p, order):
    N = len(g)
    total = 0.0
    for k, c in enumerate(order):
        I = 0.0
        row = A[c]
        for m in range(k + 1, N):
            d = order[m]
            I += row[d] * p[d]
        total += math.log2(1.0 + g[c] * p[c] / (n[c] + I))
    return total


def iterate_wf(g, n, A, P, order, iters=60):
    """Fixed-point loop: for a FIXED decode order, alternate (a) recompute each
    line's effective noise = raw noise + interference still uncancelled at its
    decode position given the current power vector, (b) re-water-fill against
    that effective-noise vector. This is interference-aware iterative
    water-filling: it converges to a self-consistent power split for the
    chosen order."""
    N = len(g)
    p = [P / N] * N
    for _ in range(iters):
        neff = [0.0] * N
        for k, c in enumerate(order):
            I = 0.0
            row = A[c]
            for m in range(k + 1, N):
                d = order[m]
                I += row[d] * p[d]
            neff[c] = n[c] + I
        p_new = waterfill(g, neff, P)
        p = [0.5 * p[i] + 0.5 * p_new[i] for i in range(N)]  # damped update
    return p


inst = json.load(sys.stdin)
N = inst["n"]
P = inst["budget"]
g = inst["gain"]
n = inst["noise"]
A = inst["coupling"]

# --- probe the interference graph: how much does line i hurt the rest, and
#     how much does it suffer from the rest? ---
out_w = [sum(A[j][i] for j in range(N)) for i in range(N)]  # harm line i causes others

candidate_orders = []

# heuristic A: decode the heaviest outgoing interferers first (get them
# cancelled out of everyone else's way as early as possible).
candidate_orders.append(sorted(range(N), key=lambda i: -out_w[i]))

# heuristic B: greedy peeling on the LIVE graph -- repeatedly pull out
# whichever remaining line is currently doing the most weighted harm to the
# lines still un-ordered (adapts as lines are removed, unlike the static
# sort above).
remaining = set(range(N))
o2 = []
while remaining:
    best = max(remaining, key=lambda i: sum(A[j][i] for j in remaining if j != i))
    o2.append(best)
    remaining.discard(best)
candidate_orders.append(o2)

# heuristic C: plain index order, as a sanity fallback.
candidate_orders.append(list(range(N)))

best_power, best_order, best_rate = None, None, float("-inf")
for order in candidate_orders:
    p = iterate_wf(g, n, A, P, order, iters=50)
    r = rate_given(g, n, A, p, order)
    if r > best_rate:
        best_rate, best_power, best_order = r, p, order

# fallback: the coupling-free water-fill + index order, so the insight-driven
# search can never end up strictly worse than the obvious approach.
def plain_waterfill(g, n, P):
    return waterfill(g, n, P)

p_fb = plain_waterfill(g, n, P)
order_fb = list(range(N))
r_fb = rate_given(g, n, A, p_fb, order_fb)
if r_fb > best_rate:
    best_rate, best_power, best_order = r_fb, p_fb, order_fb

print(json.dumps({"power": best_power, "order": best_order}))
