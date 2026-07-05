# TIER: greedy
import sys

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    D = int(d[idx]); idx += 1

    W = [0] * m
    pairs_of = [None] * m
    for ci in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        pr = []
        for _ in range(k):
            v = int(d[idx]); a = int(d[idx + 1]); idx += 2
            pr.append((v, a))
        W[ci] = w
        pairs_of[ci] = pr

    # occ[v] = list of (clause_index, demanded_config a)
    occ = [[] for _ in range(n + 1)]
    for ci in range(m):
        for (v, a) in pairs_of[ci]:
            occ[v].append((ci, a))

    # start all-default (x=0). sat[c] = # demands currently satisfied in clause c.
    x = [0] * (n + 1)
    sat = [0] * m
    for ci in range(m):
        c = 0
        for (v, a) in pairs_of[ci]:
            if x[v] == a:
                c += 1
        sat[ci] = c

    # single coordinate-ascent pass: for each module choose the config maximizing cleared weight.
    for v in range(1, n + 1):
        cur = x[v]
        # met-weight from clauses touching v, as a function of candidate config for v
        best_val = cur
        best_gain = 0  # relative to keeping cur
        # precompute per-clause "base met by others" contribution
        cand = {}
        for cfg in range(D):
            cand[cfg] = 0
        for (ci, a) in occ[v]:
            others = sat[ci] - (1 if x[v] == a else 0)  # demands met by variables other than v
            w = W[ci]
            if others > 0:
                # clause already met regardless of v -> contributes w to every candidate
                for cfg in range(D):
                    cand[cfg] += w
            else:
                # only this v can meet it, and only if x[v] == a
                cand[a] += w
        for cfg in range(D):
            gain = cand[cfg] - cand[cur]
            if gain > best_gain:
                best_gain = gain
                best_val = cfg
        if best_val != cur:
            # apply reconfiguration
            for (ci, a) in occ[v]:
                if a == cur:
                    sat[ci] -= 1
                if a == best_val:
                    sat[ci] += 1
            x[v] = best_val

    sys.stdout.write(" ".join(str(x[v]) for v in range(1, n + 1)) + "\n")

main()
