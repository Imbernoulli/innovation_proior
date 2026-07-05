# TIER: strong
# Agglomerative region merging (correlation-clustering style).  Start with every
# cell in its own zone, then repeatedly merge the pair of orthogonally adjacent
# zones whose merge reduces the TOTAL cost the most:
#     net = [SSE(A|B) - SSE(A) - SSE(B)]        # mismatch always grows on merge
#           - wall_penalty * (# adjacent A-B cell pairs)   # walls removed
# Merge while any net < 0.  This directly optimizes the real objective and finds a
# balanced partition -- far fewer walls than the salt-and-pepper bucketing, far
# less mismatch than one zone -- yet the unreachable q_lb=0 keeps it below 1.0.
import sys, json

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]
wp = inst["wall_penalty"]
flat = [t for row in inst["ideal"] for t in row]
N = H * W

# adjacency list of cell pairs (undirected)
adj = []
for i in range(H):
    for j in range(W):
        c = i * W + j
        if j + 1 < W:
            adj.append((c, c + 1))
        if i + 1 < H:
            adj.append((c, c + W))

label = list(range(N))            # each cell its own zone


def sse(cnt, s, ss):
    if cnt <= 0:
        return 0.0
    return ss - (s * s) / cnt


while True:
    # current zone stats
    cnt = {}
    ssum = {}
    ssq = {}
    for idx in range(N):
        L = label[idx]
        t = flat[idx]
        cnt[L] = cnt.get(L, 0) + 1
        ssum[L] = ssum.get(L, 0) + t
        ssq[L] = ssq.get(L, 0) + t * t
    # cross-edge counts between distinct adjacent zones
    cross = {}
    for (a, b) in adj:
        la = label[a]
        lb = label[b]
        if la != lb:
            key = (la, lb) if la < lb else (lb, la)
            cross[key] = cross.get(key, 0) + 1
    best = None
    best_net = -1e-9
    for (la, lb), ce in cross.items():
        s = ssum[la] + ssum[lb]
        c = cnt[la] + cnt[lb]
        q = ssq[la] + ssq[lb]
        merged_sse = sse(c, s, q)
        delta_mismatch = merged_sse - sse(cnt[la], ssum[la], ssq[la]) - sse(cnt[lb], ssum[lb], ssq[lb])
        net = delta_mismatch - wp * ce
        if net < best_net:
            best_net = net
            best = (la, lb)
    if best is None:
        break
    la, lb = best
    for idx in range(N):
        if label[idx] == la:
            label[idx] = lb

print(json.dumps({"labels": label}))
