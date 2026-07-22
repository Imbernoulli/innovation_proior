# TIER: strong
# Insight: at a LOW quantile an item's marginal worth depends on which scenarios
# your CURRENT bundle already loses.  So do not rank items by their own mean;
# rank by how much they lift the tail of the bundle you have so far.  Guide the
# fill by the CVaR-style tail sum (the lowest-k scenario totals) -- this hedges
# automatically, pulling in anti-correlated LO items exactly when the HI-heavy
# bundle is exposed to the low-f market crash.  Portfolio view, not knapsack.
import sys


def main():
    data = sys.stdin.buffer.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        x = data[idx]; idx += 1
        return int(x)

    N = nxt(); S = nxt(); W = nxt(); V = nxt(); k = nxt()
    w = [0] * N; v = [0] * N; vals = [None] * N
    for i in range(N):
        w[i] = nxt(); v[i] = nxt()
        vals[i] = [nxt() for _ in range(S)]

    tot = [0] * S
    remaining = set(range(N))
    cw = cv = 0
    sel = []

    def tail_sum(arr):
        # sum of the k smallest entries
        return sum(sorted(arr)[:k])

    cur_tail = tail_sum(tot)  # = 0
    while True:
        best_i = -1
        best_gain = 0.0
        for i in remaining:
            if cw + w[i] > W or cv + v[i] > V:
                continue
            row = vals[i]
            # candidate tail sum after adding item i
            ns = tail_sum([tot[s] + row[s] for s in range(S)])
            gain = (ns - cur_tail) / float(w[i] + v[i])
            if gain > best_gain:
                best_gain = gain; best_i = i
        if best_i < 0:
            break
        i = best_i
        row = vals[i]
        for s in range(S):
            tot[s] += row[s]
        cur_tail = tail_sum(tot)
        cw += w[i]; cv += v[i]
        sel.append(i); remaining.discard(i)

    sys.stdout.write(" ".join(str(i) for i in sel) + "\n")


main()
