# TIER: greedy
import sys, json


def waterfill(g, n, P):
    """Textbook water-filling: maximize sum log2(1+g_i*p_i/n_i) s.t. sum p_i<=P.
    Treats every line as an ISOLATED AWGN channel -- ignores crosstalk entirely."""
    N = len(g)
    lo, hi = 0.0, max(n[i] / g[i] for i in range(N)) + P + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = sum(max(0.0, mid - n[i] / g[i]) for i in range(N))
        if s > P:
            hi = mid
        else:
            lo = mid
    mu = (lo + hi) / 2.0
    p = [max(0.0, mu - n[i] / g[i]) for i in range(N)]
    tot = sum(p)
    if tot > P and tot > 1e-12:
        scale = P / tot
        p = [x * scale for x in p]
    return p


inst = json.load(sys.stdin)
N = inst["n"]
P = inst["budget"]
g = inst["gain"]
n = inst["noise"]
# The "coupling" matrix is in the input but this obvious first approach never
# looks at it: solve the classic single-user water-filling problem as if the
# N lines were independent AWGN channels, then decode in plain index order.
power = waterfill(g, n, P)
order = list(range(N))

print(json.dumps({"power": power, "order": order}))
