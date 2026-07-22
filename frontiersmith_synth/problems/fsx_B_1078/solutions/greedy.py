# TIER: greedy
"""The obvious approach: sort tasks by profit descending, and greedily first-fit
each task onto the first production line it can be *appended* to (respecting the
per-line horizon H and precedence order against every task already on that line).
This is exactly the trap the family targets: it cannot see that a handful of
individually high-profit, mutually-incomparable tasks each permanently occupy a
whole line (width-1 antichain members), starving the lines that could otherwise
carry a much deeper, much more profitable precedence chain."""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    N = nxt(); M = nxt(); k = nxt(); H = nxt()
    profit = [0] * (N + 1)
    for i in range(1, N + 1):
        profit[i] = nxt()
    adj = [[] for _ in range(N + 1)]
    indeg = [0] * (N + 1)
    for _ in range(M):
        u = nxt(); v = nxt()
        adj[u].append(v)
        indeg[v] += 1

    dq = deque([i for i in range(1, N + 1) if indeg[i] == 0])
    topo = []
    indeg2 = indeg[:]
    while dq:
        u = dq.popleft()
        topo.append(u)
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                dq.append(v)

    reach = [0] * (N + 1)
    for u in reversed(topo):
        r = 0
        for w in adj[u]:
            r |= (1 << w) | reach[w]
        reach[u] = r

    def reachable(a, b):
        return (reach[a] >> b) & 1 == 1

    order = sorted(range(1, N + 1), key=lambda t: (-profit[t], t))

    lines = [[] for _ in range(k)]
    for t in order:
        for ln in lines:
            if len(ln) >= H:
                continue
            if all(reachable(prev, t) for prev in ln):
                ln.append(t)
                break

    used_lines = [ln for ln in lines if ln]
    out = [str(len(used_lines))]
    for ln in used_lines:
        out.append(str(len(ln)) + " " + " ".join(map(str, ln)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
