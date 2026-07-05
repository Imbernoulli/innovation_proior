# TIER: strong
# Graves-Willems dynamic program: the exact optimal service-time placement on the
# tree. Define f_i(s_in) = minimum safety-stock cost of the subtree rooted at i given
# that i's inbound service time is s_in. Node i may quote any integer outbound service
# S_i in [0, min(s_in + T_i, cap_i)] (cap_i = s_max for a station, unbounded else);
# it then pays h_i*k*sigma_i*sqrt(s_in + T_i - S_i) and hands S_i down as the inbound
# service of every child:
#     f_i(s_in) = min_{S_i} [ h_i*k*sigma_i*sqrt(s_in+T_i-S_i) + sum_children f_c(S_i) ]
# Process nodes deepest-first (leaves first), then read the optimum off the root at
# s_in = Sext and reconstruct the argmin service times top-down. This trades pooling
# (sqrt lead-time savings) against the downstream-rising holding cost and the station
# caps, and is far cheaper than the decoupled baseline while never hitting a trivial
# optimum on the harder instances.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]; parent = inst["parent"]; T = inst["T"]; h = inst["h"]
sigma = inst["sigma"]; smax = inst["smax"]; level = inst["level"]
Sext = inst["Sext"]; k = inst["k"]
ch = inst.get("children")
if ch is None:
    ch = [[] for _ in range(n)]
    for i in range(1, n):
        ch[parent[i]].append(i)

# tightest useful service-time bound: cumulative lead time along the path from root
cum = [0] * n
for i in sorted(range(n), key=lambda i: level[i]):
    cum[i] = (Sext if parent[i] < 0 else cum[parent[i]]) + T[i]
Smax = max(cum) if n else 0

f = [None] * n          # f[i][s_in] = (min_cost, best_Si)
for i in sorted(range(n), key=lambda i: -level[i]):
    row = [None] * (Smax + 1)
    base_i = h[i] * k * sigma[i]
    cap_i = smax[i]
    kids = ch[i]
    for s_in in range(Smax + 1):
        hi = s_in + T[i]
        if hi > Smax:
            hi = Smax
        cap = hi if hi < cap_i else cap_i
        best = None; bS = 0
        for Si in range(cap + 1):
            c = base_i * math.sqrt(s_in + T[i] - Si)
            for cc in kids:
                c += f[cc][Si][0]
            if best is None or c < best:
                best = c; bS = Si
        row[s_in] = (best, bS)
    f[i] = row

S = [0] * n
stack = [(0, Sext)]
while stack:
    i, s_in = stack.pop()
    Si = f[i][s_in][1]
    S[i] = Si
    for cc in ch[i]:
        stack.append((cc, Si))
print(json.dumps({"S": S}))
