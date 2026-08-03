# TIER: greedy
import sys

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    clauses = []
    for _ in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        lits = [int(d[idx + j]) for j in range(k)]
        idx += k
        clauses.append((w, lits))

    # occ[v] = list of (clause_index, is_positive_literal)
    occ = [[] for _ in range(n + 1)]
    for ci, (w, lits) in enumerate(clauses):
        for l in lits:
            occ[abs(l)].append((ci, l > 0))

    # start all retrograde (x=0). num_true[c] = # literals currently satisfied.
    x = [0] * (n + 1)
    num_true = [0] * m
    for ci, (w, lits) in enumerate(clauses):
        cnt = 0
        for l in lits:
            if (l > 0 and x[abs(l)] == 1) or (l < 0 and x[abs(l)] == 0):
                cnt += 1
        num_true[ci] = cnt

    W = [w for (w, _) in clauses]

    # single greedy pass: for each satellite, adopt the mode that increases cleared weight.
    for v in range(1, n + 1):
        # delta of flipping x[v]
        delta = 0
        newv = 1 - x[v]
        for (ci, pos) in occ[v]:
            # literal at v currently true?
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true == aft_true:
                continue
            nt = num_true[ci]
            if cur_true and not aft_true:
                # losing a true literal
                if nt == 1:
                    delta -= W[ci]
            else:
                # gaining a true literal
                if nt == 0:
                    delta += W[ci]
        if delta > 0:
            # apply flip
            for (ci, pos) in occ[v]:
                cur_true = (x[v] == 1) == pos
                aft_true = (newv == 1) == pos
                if cur_true and not aft_true:
                    num_true[ci] -= 1
                elif aft_true and not cur_true:
                    num_true[ci] += 1
            x[v] = newv

    sys.stdout.write(" ".join(str(x[v]) for v in range(1, n + 1)) + "\n")

main()
