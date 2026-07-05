# TIER: greedy
import sys

def parse(d):
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    D = int(d[idx]); idx += 1
    W = [0] * m
    lits = [None] * m
    for c in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        ll = []
        for _ in range(k):
            v = int(d[idx]); a = int(d[idx + 1]); idx += 2
            ll.append((v, a))
        W[c] = w
        lits[c] = ll
    return n, m, D, W, lits

def main():
    d = sys.stdin.buffer.read().split()
    n, m, D, W, lits = parse(d)

    # occ[v] = list of (clause, required_channel_for_v)
    occ = [[] for _ in range(n + 1)]
    for c in range(m):
        for (v, a) in lits[c]:
            occ[v].append((c, a))

    # start all channel 1
    x = [1] * (n + 1)
    num_true = [0] * m
    for c in range(m):
        cnt = 0
        for (v, a) in lits[c]:
            if x[v] == a:
                cnt += 1
        num_true[c] = cnt

    # single greedy pass: coordinate-optimal channel for each station in index order
    for v in range(1, n + 1):
        aold = x[v]
        # bonus[a] = extra cleared weight from clauses currently dark (base==0)
        # whose v-literal wants channel a
        const_part = 0
        bonus = [0] * (D + 1)
        for (c, a) in occ[v]:
            base = num_true[c] - (1 if aold == a else 0)
            if base > 0:
                const_part += W[c]
            else:
                bonus[a] += W[c]
        best_a = 1
        best_b = bonus[1]
        for a in range(2, D + 1):
            if bonus[a] > best_b:
                best_b = bonus[a]; best_a = a
        if best_a != aold:
            for (c, a) in occ[v]:
                if aold == a:
                    num_true[c] -= 1
                if best_a == a:
                    num_true[c] += 1
            x[v] = best_a

    sys.stdout.write(" ".join(str(x[v]) for v in range(1, n + 1)) + "\n")

main()
