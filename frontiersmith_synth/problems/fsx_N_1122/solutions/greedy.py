# TIER: greedy
"""
The "obvious" first pass: treat this like textbook influence-maximization --
fire every source (in-degree-0 node) at time 0 (the classic static seed
set), simulate, and directly patch whatever is still inactive at the end.
It never considers OFFSETTING sources in time, so any join that needs its
two incoming pulses to land in the same simulation step (their sum clears
the threshold, but neither pulse alone does, and an unmatched pulse decays
away for good) is missed and has to be patched directly -- one extra event
per such join.
"""
import sys


def read_instance(toks):
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); m = int(nxt())
    decay_num = int(nxt()); decay_den = int(nxt())
    ext_boost = int(nxt()); horizon = int(nxt())
    theta = [int(nxt()) for _ in range(n)]
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for _ in range(m):
        u = int(nxt()); v = int(nxt()); w = int(nxt())
        adj[u].append((v, w))
        indeg[v] += 1
    return n, decay_num, decay_den, ext_boost, horizon, theta, adj, indeg


def simulate(n, theta, adj, decay_num, decay_den, ext_boost, horizon, events):
    acc = [0] * n
    active = [False] * n
    just_activated = [False] * n
    for t in range(horizon):
        for node in events.get(t, []):
            if not active[node]:
                acc[node] += ext_boost
        contrib = [0] * n
        for u in range(n):
            if just_activated[u]:
                for (v, w) in adj[u]:
                    if not active[v]:
                        contrib[v] += w
        for v in range(n):
            if contrib[v] and not active[v]:
                acc[v] += contrib[v]
        newly = []
        for v in range(n):
            if not active[v] and acc[v] >= theta[v]:
                newly.append(v)
        just_activated = [False] * n
        for v in newly:
            active[v] = True
            just_activated[v] = True
        for v in range(n):
            if not active[v]:
                acc[v] = (acc[v] * decay_num) // decay_den
    return active


def main():
    toks = sys.stdin.read().split()
    n, decay_num, decay_den, ext_boost, horizon, theta, adj, indeg = read_instance(toks)

    events = {0: [v for v in range(n) if indeg[v] == 0]}
    active = simulate(n, theta, adj, decay_num, decay_den, ext_boost, horizon, events)

    schedule = [(v, 0) for v in range(n) if indeg[v] == 0]
    patch_time = horizon - 1
    for v in range(n):
        if not active[v]:
            events.setdefault(patch_time, []).append(v)
            schedule.append((v, patch_time))
            active[v] = True  # ext_boost always clears any threshold instantly

    out = [str(len(schedule))]
    for (v, t) in schedule:
        out.append(f"{v} {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
