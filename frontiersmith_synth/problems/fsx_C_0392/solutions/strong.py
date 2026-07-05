# TIER: strong
# Constraint-based (PC-style) causal discovery.
#   1. Skeleton: start from all pairs whose marginal mutual information exceeds a
#      threshold.
#   2. Order-1 conditional-independence pruning: remove edge X-Y if there is a
#      neighbour Z such that the conditional mutual information CMI(X;Y|Z) drops
#      below a (bias-corrected) threshold -- this kills the spurious transitive
#      links (A-C explained by the mediator B) that pure association keeps, and
#      records Z in the separating set of {X,Y}.
#   3. Orientation: apply the v-structure (collider) rule -- for every unshielded
#      triple X - Z - Y with X,Y non-adjacent and Z NOT in sepset(X,Y), orient
#      X -> Z <- Y.  Then a light Meek-style propagation orients further edges to
#      avoid new colliders / cycles; any still-undirected edge is oriented by a
#      deterministic degree tiebreak.
# Getting the skeleton right (few spurious links) plus partially-correct
# orientation drives SHD well below the empty-graph baseline, but Markov-
# equivalence and finite samples keep it above zero -> headroom remains.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
card = inst["card"]
data = inst["data"]
m = len(data)

cols = [[row[i] for row in data] for i in range(n)]

MI_THR = 0.010          # skeleton inclusion (nats)
CMI_BASE = 0.005        # order-1 removal, plus a per-df bias margin


def mutual_info(i, j):
    joint = {}
    ai = {}
    bj = {}
    for a, b in zip(cols[i], cols[j]):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        ai[a] = ai.get(a, 0) + 1
        bj[b] = bj.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        pab = c / m
        mi += pab * math.log(pab / ((ai[a] / m) * (bj[b] / m)))
    return mi


def cmi(i, j, k):
    # conditional mutual information CMI(i;j|k), nats
    cnt_abz = {}
    cnt_az = {}
    cnt_bz = {}
    cnt_z = {}
    ci, cj, ck = cols[i], cols[j], cols[k]
    for a, b, z in zip(ci, cj, ck):
        cnt_abz[(a, b, z)] = cnt_abz.get((a, b, z), 0) + 1
        cnt_az[(a, z)] = cnt_az.get((a, z), 0) + 1
        cnt_bz[(b, z)] = cnt_bz.get((b, z), 0) + 1
        cnt_z[z] = cnt_z.get(z, 0) + 1
    val = 0.0
    for (a, b, z), c in cnt_abz.items():
        pabz = c / m
        num = c * cnt_z[z]
        den = cnt_az[(a, z)] * cnt_bz[(b, z)]
        val += pabz * math.log(num / den)
    return val


# ---- step 1: skeleton from marginal MI ----
adj = [set() for _ in range(n)]
mi_cache = {}
for i in range(n):
    for j in range(i + 1, n):
        v = mutual_info(i, j)
        mi_cache[(i, j)] = v
        if v > MI_THR:
            adj[i].add(j)
            adj[j].add(i)

# ---- step 2: order-1 CI pruning ----
sepset = {}
edge_list = [(i, j) for i in range(n) for j in adj[i] if i < j]
for (i, j) in edge_list:
    if j not in adj[i]:
        continue
    conds = (adj[i] | adj[j]) - {i, j}
    removed = False
    for k in sorted(conds):
        # bias margin grows with the conditioning cardinality and df
        df = (card[i] - 1) * (card[j] - 1) * card[k]
        thr = CMI_BASE + df / (2.0 * m)
        if cmi(i, j, k) < thr:
            adj[i].discard(j)
            adj[j].discard(i)
            key = (i, j) if i < j else (j, i)
            sepset[key] = k
            removed = True
            break
    # (if not removed the edge stays)

# ---- step 3a: v-structure orientation ----
# directed[(a,b)] means a -> b
directed = set()


def adjacent(a, b):
    return b in adj[a]


for z in range(n):
    nb = sorted(adj[z])
    for ii in range(len(nb)):
        for jj in range(ii + 1, len(nb)):
            x, y = nb[ii], nb[jj]
            if adjacent(x, y):
                continue                     # shielded, not a collider triple
            key = (x, y) if x < y else (y, x)
            if sepset.get(key, -1) != z:     # z not in sepset(x,y) -> collider
                directed.add((x, z))
                directed.add((y, z))

# ---- step 3b: light Meek-style propagation (avoid new colliders & cycles) ----
def has_path(a, b, banned):
    # directed path a ->...-> b using current `directed`, avoiding edge `banned`
    stack = [a]
    seen = {a}
    while stack:
        u = stack.pop()
        for w in range(n):
            if (u, w) in directed and (u, w) != banned and w not in seen:
                if w == b:
                    return True
                seen.add(w)
                stack.append(w)
    return False


changed = True
while changed:
    changed = False
    # Meek R1: if a->b and b-c (undirected) and a,c non-adjacent -> b->c
    for (a, b) in list(directed):
        for c in sorted(adj[b]):
            if c == a:
                continue
            if (b, c) in directed or (c, b) in directed:
                continue
            if not adjacent(a, c):
                directed.add((b, c))
                changed = True
    # Meek R2: if a->c->b and a-b undirected -> a->b
    for a in range(n):
        for b in sorted(adj[a]):
            if (a, b) in directed or (b, a) in directed:
                continue
            for c in range(n):
                if (a, c) in directed and (c, b) in directed:
                    directed.add((a, b))
                    changed = True
                    break

# ---- step 3c: orient any remaining undirected edges by a deterministic rule ----
# orient from lower-degree endpoint to higher-degree (ties: lower index -> higher),
# skipping if it would create a directed cycle.
deg = [len(adj[i]) for i in range(n)]
for i in range(n):
    for j in sorted(adj[i]):
        if i >= j:
            continue
        if (i, j) in directed or (j, i) in directed:
            continue
        # choose orientation a->b
        if (deg[i], i) <= (deg[j], j):
            a, b = i, j
        else:
            a, b = j, i
        if has_path(b, a, banned=(a, b)):    # would close a cycle -> flip
            a, b = b, a
        directed.add((a, b))

# emit: one directed edge per surviving skeleton pair
out = []
emitted = set()
for i in range(n):
    for j in adj[i]:
        if i < j:
            if (i, j) in directed:
                pair = (i, j)
            elif (j, i) in directed:
                pair = (j, i)
            else:
                pair = (i, j)
            if pair not in emitted:
                emitted.add(pair)
                out.append([pair[0], pair[1]])

print(json.dumps({"edges": out}))
