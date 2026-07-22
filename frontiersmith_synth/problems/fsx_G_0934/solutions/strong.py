# TIER: strong
# RPNI-style EVIDENCE-BASED STATE MERGING. Build the prefix-tree acceptor (same tree
# as the greedy solution), then walk its states in standard order (shortest / then
# lexicographically first) and, for each new state, try folding it into the EARLIEST
# already-processed state that keeps the automaton consistent with every labeled
# training example. A candidate merge is accepted only if it never forces an
# evidenced-accept prefix and an evidenced-reject prefix into the same state; merging
# two states forces their same-symbol children to merge too (propagated via a
# union-find with rollback-on-conflict). What survives is a SMALL automaton that
# treats prefixes as evidence for a shared underlying device state, instead of a
# separate memorized branch per training string -- this is what recovers behavior on
# traces the prefix tree never saw.
import sys, json

inst = json.load(sys.stdin)
train = inst["train"]

# ---- build the PTA's prefix set / standard order / raw tree transitions ----
prefixes = {""}
for t in train:
    s = t["s"]
    for i in range(1, len(s) + 1):
        prefixes.add(s[:i])
order = sorted(prefixes, key=lambda p: (len(p), p))
idx = {p: i for i, p in enumerate(order)}
n = len(order)

lit_label = {}
for t in train:
    lit_label[t["s"]] = t["label"]

label = [lit_label.get(order[i]) for i in range(n)]  # 0/1/None per PTA state

trans = [dict() for _ in range(n)]  # trans[i][0/1] = child idx, only if child exists
for p in prefixes:
    i = idx[p]
    for sym, c in ((0, "0"), (1, "1")):
        q = p + c
        if q in prefixes:
            trans[i][sym] = idx[q]

# ---- union-find with per-block label + per-block outgoing-transition witness ----
parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


root_label = {i: label[i] for i in range(n) if label[i] is not None}
root_out = {i: dict(trans[i]) for i in range(n)}


def try_merge(a, b):
    """Attempt to fold the blocks containing original states a, b (a is the
    'earlier'/kept side). Commit and return True iff consistent with every label."""
    snap_parent = parent[:]
    snap_label = dict(root_label)
    snap_out = {k: dict(v) for k, v in root_out.items()}

    queue = [(a, b)]
    ok = True
    while queue:
        x, y = queue.pop()
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        lx, ly = root_label.get(rx), root_label.get(ry)
        if lx is not None and ly is not None and lx != ly:
            ok = False
            break
        parent[ry] = rx
        if lx is None and ly is not None:
            root_label[rx] = ly
        root_label.pop(ry, None)
        ox = root_out.get(rx, {})
        oy = root_out.pop(ry, {})
        for sym, tgt in oy.items():
            if sym in ox:
                queue.append((ox[sym], tgt))
            else:
                ox[sym] = tgt
        root_out[rx] = ox

    if not ok:
        parent[:] = snap_parent
        root_label.clear(); root_label.update(snap_label)
        root_out.clear(); root_out.update(snap_out)
        return False
    return True


for i in range(1, n):
    for j in range(0, i):
        if find(j) == find(i):
            break
        if try_merge(j, i):
            break

# ---- compact the surviving blocks into the output automaton ----
final_roots = sorted(set(find(i) for i in range(n)))
comp_id = {r: k for k, r in enumerate(final_roots)}
m = len(final_roots)
out_delta = [[None, None] for _ in range(m)]
out_accept = set()
for r in final_roots:
    ci = comp_id[r]
    if root_label.get(r) == 1:
        out_accept.add(ci)
    for sym in (0, 1):
        tgt = root_out.get(r, {}).get(sym)
        if tgt is not None:
            out_delta[ci][sym] = comp_id[find(tgt)]

missing = any(out_delta[ci][sym] is None for ci in range(m) for sym in (0, 1))
if missing:
    SINK = m
    for ci in range(m):
        for sym in (0, 1):
            if out_delta[ci][sym] is None:
                out_delta[ci][sym] = SINK
    out_delta.append([SINK, SINK])
    m += 1

start = comp_id[find(0)]
print(json.dumps({"delta": out_delta, "start": start, "accept": sorted(out_accept)}))
