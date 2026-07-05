# TIER: greedy
import sys

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    W = [0] * m
    T = [0] * m
    lit_lists = [None] * m
    for ci in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        t = int(d[idx]); idx += 1
        lits = [int(d[idx + j]) for j in range(k)]
        idx += k
        W[ci] = w; T[ci] = t; lit_lists[ci] = lits

    # occ[v] = list of (zone_index, is_positive_literal)
    occ = [[] for _ in range(n + 1)]
    for ci in range(m):
        for l in lit_lists[ci]:
            occ[abs(l)].append((ci, l > 0))

    # start all production (x=0). c[ci] = # satisfied literals currently.
    x = [0] * (n + 1)
    c = [0] * m
    for ci in range(m):
        cnt = 0
        for l in lit_lists[ci]:
            if (l > 0 and x[abs(l)] == 1) or (l < 0 and x[abs(l)] == 0):
                cnt += 1
        c[ci] = cnt

    # single greedy pass: for each well, adopt the mode that increases cleared weight.
    for v in range(1, n + 1):
        newv = 1 - x[v]
        delta = 0
        for (ci, pos) in occ[v]:
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true == aft_true:
                continue
            if cur_true and not aft_true:
                if c[ci] == T[ci]:
                    delta -= W[ci]
            else:
                if c[ci] == T[ci] - 1:
                    delta += W[ci]
        if delta > 0:
            for (ci, pos) in occ[v]:
                cur_true = (x[v] == 1) == pos
                aft_true = (newv == 1) == pos
                if cur_true and not aft_true:
                    c[ci] -= 1
                elif aft_true and not cur_true:
                    c[ci] += 1
            x[v] = newv

    sys.stdout.write(" ".join(str(x[v]) for v in range(1, n + 1)) + "\n")

main()
