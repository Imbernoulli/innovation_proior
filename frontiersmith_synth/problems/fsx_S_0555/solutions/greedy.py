# TIER: greedy
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    alphabet = next(it)
    nP = int(next(it)); nN = int(next(it))
    P = [next(it) for _ in range(nP)]
    N = [next(it) for _ in range(nN)]
    A = list(alphabet); ci = {c: i for i, c in enumerate(A)}

    # trie DFA over exact P + dead sink, then Moore minimization
    pref = set()
    for s in P:
        for i in range(len(s) + 1):
            pref.add(s[:i])
    states = sorted(pref)
    idx = {s: i for i, s in enumerate(states)}
    DEAD = len(states)
    n = len(states) + 1
    accept = [False] * n
    for s in P:
        accept[idx[s]] = True
    delta = [[DEAD] * len(A) for _ in range(n)]
    for s in pref:
        for c in A:
            t = s + c
            if t in idx:
                delta[idx[s]][ci[c]] = idx[t]

    part = [1 if accept[i] else 0 for i in range(n)]
    while True:
        keys = {}; newp = [0] * n; nx = 0
        for i in range(n):
            k = (part[i], tuple(part[delta[i][c]] for c in range(len(A))))
            if k not in keys:
                keys[k] = nx; nx += 1
            newp[i] = keys[k]
        if newp == part:
            break
        part = newp

    start_state = idx[""]
    # reachable original states
    reach = set(); st = [start_state]
    while st:
        u = st.pop()
        if u in reach:
            continue
        reach.add(u)
        for c in range(len(A)):
            st.append(delta[u][c])
    dead_class = part[DEAD]
    # emit classes reachable, excluding the pure dead sink class
    class_ids = {}
    def cid(cl):
        if cl not in class_ids:
            class_ids[cl] = len(class_ids)
        return class_ids[cl]
    # order: ensure start class gets an id
    start_cl = part[start_state]
    cid(start_cl)
    edges = []
    acc_classes = set()
    seen_cls = set()
    for u in reach:
        cl = part[u]
        if cl == dead_class:
            continue
        if cl in seen_cls:
            continue
        seen_cls.add(cl)
        _ = cid(cl)
        if accept[u]:
            acc_classes.add(cl)
        for c in range(len(A)):
            tcl = part[delta[u][c]]
            if tcl == dead_class:
                continue
            edges.append((cl, A[c], tcl))
    # renumber
    S = len(class_ids)
    remap = class_ids
    E = []
    eset = set()
    for (f, sym, t) in edges:
        key = (remap[f], sym, remap[t])
        if key in eset:
            continue
        eset.add(key)
        E.append(key)
    starts = [remap[start_cl]]
    acc = sorted(remap[c] for c in acc_classes)
    out = []
    out.append(str(S))
    out.append(str(len(starts)) + " " + " ".join(map(str, starts)))
    out.append(str(len(acc)) + " " + " ".join(map(str, acc)))
    out.append(str(len(E)))
    for (f, sym, t) in E:
        out.append("%d %s %d" % (f, sym, t))
    sys.stdout.write("\n".join(out) + "\n")

main()
