# TIER: strong
# RPNI-style grammar induction (regular-language state merging).
#   1. Build the prefix-tree acceptor (PTA) from the labeled training trellises.
#   2. In shortlex (BFS) order, repeatedly try to MERGE each blue-fringe state into an
#      already-confirmed (red) state; a merge is folded to keep the automaton
#      deterministic and is rejected if it forces an accepting state to coincide with a
#      rejecting one.  If no red merge is consistent, promote the blue state to red.
#   3. The quotient automaton (plus a rejecting sink for unseen transitions) is the
#      induced rule.
# By recovering the underlying finite-state STRUCTURE rather than memorizing strings,
# this generalizes to the much taller OOD trellises -- but exact identification from
# finite data is under-determined, so it stays below a perfect score on the sparser /
# larger instances.  Falls back to a majority DFA if anything goes wrong.
import sys, json

inst = json.load(sys.stdin)
D = inst["n_types"]
M = inst["max_states"]
train = inst["train"]

ones = sum(1 for _, y in train if y == 1)
gmaj = 1 if ones * 2 >= len(train) else 0


def majority_dfa():
    return {"start": 0, "accept": [gmaj], "trans": [[0] * D]}


def induce():
    # ---- build PTA ----
    children = [dict()]      # children[node][sym] = node
    label = [-1]             # -1 unknown / 0 / 1
    for seq, y in train:
        u = 0
        for c in seq:
            nxt = children[u].get(c)
            if nxt is None:
                nxt = len(children)
                children.append(dict())
                label.append(-1)
                children[u][c] = nxt
            u = nxt
        label[u] = y
    Q = len(children)

    # ---- shortlex BFS rank ----
    rank = [0] * Q
    order = []
    seen = [False] * Q
    queue = [0]
    seen[0] = True
    while queue:
        u = queue.pop(0)
        order.append(u)
        for c in range(D):
            v = children[u].get(c)
            if v is not None and not seen[v]:
                seen[v] = True
                queue.append(v)
    for i, u in enumerate(order):
        rank[u] = i

    # ---- class structure (union-find; rep = lowest rank in class) ----
    parent = list(range(Q))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    classlabel = list(label)          # by rep
    childmap = [dict(children[u]) for u in range(Q)]   # by rep: sym -> node

    def snapshot():
        return (parent[:], classlabel[:],
                [d.copy() if d is not None else None for d in childmap])

    def restore(snap):
        p, cl, cm = snap
        parent[:] = p
        classlabel[:] = cl
        for i in range(Q):
            childmap[i] = cm[i]

    def apply_merge(q, r):
        stack = [(q, r)]
        while stack:
            x, y = stack.pop()
            a = find(x)
            b = find(y)
            if a == b:
                continue
            lo, hi = (a, b) if rank[a] <= rank[b] else (b, a)
            la, lb = classlabel[lo], classlabel[hi]
            if la != -1 and lb != -1 and la != lb:
                return False
            parent[hi] = lo
            classlabel[lo] = la if la != -1 else lb
            hm = childmap[hi]
            lm = childmap[lo]
            for sym, tgt in hm.items():
                if sym in lm:
                    stack.append((lm[sym], tgt))
                else:
                    lm[sym] = tgt
            childmap[hi] = None
        return True

    # ---- RPNI main loop ----
    red = [0]
    red_set = {0}

    def compute_blue():
        cand = set()
        for r in red:
            rr = find(r)
            cm = childmap[rr]
            if cm is None:
                continue
            for c in range(D):
                if c in cm:
                    t = find(cm[c])
                    if t not in red_set:
                        cand.add(t)
        return sorted(cand, key=lambda z: rank[z])

    guard = 0
    while True:
        guard += 1
        if guard > Q + 5:
            break
        blue = compute_blue()
        if not blue:
            break
        q = blue[0]
        merged = False
        for r in list(red):
            rr = find(r)
            qq = find(q)
            if rr == qq:
                merged = True
                break
            snap = snapshot()
            if apply_merge(qq, rr):
                merged = True
                break
            else:
                restore(snap)
        if not merged:
            red.append(find(q))
            red_set = {find(x) for x in red}
        else:
            red = [find(x) for x in red]
            red_set = set(red)
        if len(red) > M - 1:
            return None      # too big -> caller falls back

    # ---- build quotient DFA (+ rejecting sink) ----
    reps = sorted({find(r) for r in red}, key=lambda z: rank[z])
    ridx = {rep: i for i, rep in enumerate(reps)}
    K = len(reps)
    sink = K
    K_total = K + 1
    if K_total > M:
        return None
    trans = [[sink] * D for _ in range(K_total)]
    accept = [0] * K_total
    for rep in reps:
        i = ridx[rep]
        cl = classlabel[rep]
        accept[i] = 1 if cl == 1 else 0
        cm = childmap[rep]
        if cm:
            for c in range(D):
                if c in cm:
                    t = find(cm[c])
                    trans[i][c] = ridx.get(t, sink)
    # sink self-loops, non-accepting
    for c in range(D):
        trans[sink][c] = sink
    accept[sink] = 0
    start = ridx[find(0)]
    return {"start": start, "accept": accept, "trans": trans}


try:
    dfa = induce()
    if dfa is None:
        dfa = majority_dfa()
except Exception:
    dfa = majority_dfa()

print(json.dumps(dfa))
