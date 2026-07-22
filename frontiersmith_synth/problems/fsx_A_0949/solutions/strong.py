# TIER: strong
import sys
from collections import deque


def simulate(N, sources, theta, amp, adj, removed):
    active = [False] * (N + 1)
    inw = [0.0] * (N + 1)
    q = deque()
    for s in sources:
        if s in removed:
            continue
        active[s] = True
        q.append(s)
    while q:
        u = q.popleft()
        au = amp[u]
        for (v, w) in adj[u]:
            if v in removed or active[v]:
                continue
            inw[v] += au * w
            if inw[v] >= theta[v] - 1e-9:
                active[v] = True
                q.append(v)
    return sum(1 for i in range(1, N + 1) if active[i]) - sum(
        1 for s in sources if active[s])


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); K = int(next(it)); S = int(next(it))
    sources = [int(next(it)) for _ in range(S)]
    theta = [0.0] * (N + 1)
    amp = [1.0] * (N + 1)
    for i in range(1, N + 1):
        theta[i] = float(next(it))
        amp[i] = float(next(it))
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(next(it)); v = int(next(it)); w = float(next(it))
        adj[u].append((v, w))

    # Insight: in this cascade model, a node's stated amplification factor
    # (amp_v > 1) is exactly what lets it re-broadcast its incoming
    # activation to many downstream neighbours at multiplied weight -- so
    # the nodes that matter for blocking the cascade are the amplifiers,
    # not whoever happens to sit next to a source. Restrict the search to
    # that (small) candidate set, and pick the budgeted subset by an exact
    # marginal-gain (resimulate-and-choose) selection: at each step, try
    # removing every remaining candidate on top of what's already removed,
    # and keep the one that lowers the simulated activation count the most.
    # This directly measures each amplifier's true downstream impact
    # (including any overlap between branches), rather than guessing from
    # a static score.
    src_set = set(sources)
    candidates = [i for i in range(1, N + 1) if i not in src_set and amp[i] > 1.0 + 1e-9]

    chosen = []
    remaining = set(candidates)
    budget = min(K, len(candidates))
    for _ in range(budget):
        best_node = None
        best_f = None
        for c in remaining:
            trial = set(chosen); trial.add(c)
            f = simulate(N, sources, theta, amp, adj, trial)
            if best_f is None or f < best_f or (f == best_f and (best_node is None or c < best_node)):
                best_f = f
                best_node = c
        chosen.append(best_node)
        remaining.discard(best_node)

    # If the amplifier count is smaller than the budget, don't leave the
    # rest unspent: mop up with the best remaining near-source nodes (the
    # same cheap move a naive solver would make first). The insight is
    # spending the budget on amplifiers *before* anything else, not
    # refusing whatever leftover budget can still buy.
    if len(chosen) < K:
        chosen_set = set(chosen)
        out_w = [0.0] * (N + 1)
        dist = [None] * (N + 1)
        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            for (v, w) in adj[u]:
                out_w[u] += w
                if dist[v] is None:
                    dist[v] = dist[u] + 1
                    q.append(v)
        fillers = [i for i in range(1, N + 1)
                   if i not in src_set and i not in chosen_set
                   and dist[i] is not None and dist[i] <= 1]
        fillers.sort(key=lambda i: (-out_w[i], i))
        for f in fillers:
            if len(chosen) >= K:
                break
            chosen.append(f)

    print(len(chosen))
    print(" ".join(str(x) for x in chosen))


if __name__ == "__main__":
    main()
