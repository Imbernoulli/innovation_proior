# TIER: strong
# Insight: the cumulative residue cap turns this from a pairwise tour problem into an
# EPOCH-PARTITIONING problem. For ANY fixed order the optimal placement of cleans is an exact
# O(N^2) shortest-path DP: cut the order into contiguous residue-feasible epochs; each epoch
# pays one startup for its first job plus its internal changeovers, and each boundary pays one
# clean. Placing a boundary right before an expensive dark->light changeover converts that
# costly transition into a cheap clean-machine startup -- absorbing the penalty into a reset we
# were going to spend anyway. We therefore optimize the ORDER against the DP cost (2-opt local
# search seeded by several nearest-neighbour tours), not the raw changeover total.
import sys

def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); R = int(next(it)); C = int(next(it))
    r = [int(next(it)) for _ in range(N)]
    s = [int(next(it)) for _ in range(N)]
    w = [[int(next(it)) for _ in range(N)] for _ in range(N)]
    return N, R, C, r, s, w

def nn_from(N, w, start):
    used = [False] * N; used[start] = True
    order = [start]; cur = start
    for _ in range(N - 1):
        nxt = min((j for j in range(N) if not used[j]), key=lambda j: w[cur][j])
        used[nxt] = True; order.append(nxt); cur = nxt
    return order

def dp_cost(N, R, C, r, s, w, order):
    pref = [0] * (N + 1)
    for k in range(N):
        pref[k + 1] = pref[k] + r[order[k]]
    INF = float("inf")
    dp = [INF] * (N + 1); dp[0] = 0
    for b in range(1, N + 1):
        for a in range(b - 1, -1, -1):
            if pref[b] - pref[a] > R:
                break
            ec = s[order[a]]
            for t in range(a, b - 1):
                ec += w[order[t]][order[t + 1]]
            cand = dp[a] + ec + (C if a > 0 else 0)
            if cand < dp[b]:
                dp[b] = cand
    return dp[N]

def dp_schedule(N, R, C, r, s, w, order):
    pref = [0] * (N + 1)
    for k in range(N):
        pref[k + 1] = pref[k] + r[order[k]]
    INF = float("inf")
    dp = [INF] * (N + 1); dp[0] = 0; par = [-1] * (N + 1)
    for b in range(1, N + 1):
        for a in range(b - 1, -1, -1):
            if pref[b] - pref[a] > R:
                break
            ec = s[order[a]]
            for t in range(a, b - 1):
                ec += w[order[t]][order[t + 1]]
            cand = dp[a] + ec + (C if a > 0 else 0)
            if cand < dp[b]:
                dp[b] = cand; par[b] = a
    segs = []; b = N
    while b > 0:
        a = par[b]; segs.append((a, b)); b = a
    segs.reverse()
    seq = []
    for idx, (a, b) in enumerate(segs):
        if idx > 0:
            seq.append("C")
        seq.extend(str(order[t]) for t in range(a, b))
    return seq

def main():
    N, R, C, r, s, w = read_instance()
    cands = [nn_from(N, w, st) for st in range(N)]
    cands.append(list(range(N)))
    cands.append(sorted(range(N), key=lambda j: s[j]))
    best = min(cands, key=lambda o: dp_cost(N, R, C, r, s, w, o))
    bc = dp_cost(N, R, C, r, s, w, best)
    # 2-opt segment-reversal local search, scored by the epoch-DP objective
    improved = True; budget = 6
    while improved and budget > 0:
        budget -= 1; improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                o = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = dp_cost(N, R, C, r, s, w, o)
                if c < bc - 1e-9:
                    bc = c; best = o; improved = True
    sys.stdout.write(" ".join(dp_schedule(N, R, C, r, s, w, best)) + "\n")

main()
