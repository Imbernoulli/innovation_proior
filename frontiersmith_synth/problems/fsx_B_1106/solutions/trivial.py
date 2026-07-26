# TIER: trivial
# Baseline construction: cover every transition with its OWN separate string --
# the shortest path (in symbols) from s0 to that transition's source state,
# followed by the transition's symbol. Always feasible, never reuses anything
# across transitions (restarts from s0 every time) -> reproduces the checker's
# internal baseline exactly.
import sys
from collections import deque


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    n, k, s0 = int(head[0]), int(head[1]), int(head[2])
    symbols = data[1].split()
    trans = []
    for i in range(n):
        trans.append([int(x) for x in data[2 + i].split()])

    dist = [-1] * n
    parent = [None] * n
    dist[s0] = 0
    dq = deque([s0])
    while dq:
        u = dq.popleft()
        for s in range(k):
            v = trans[u][s]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = (u, s)
                dq.append(v)

    def path_str(target):
        if target == s0:
            return ""
        seq = []
        cur = target
        while cur != s0:
            u, s = parent[cur]
            seq.append(symbols[s])
            cur = u
        seq.reverse()
        return "".join(seq)

    lines = [str(n * k)]
    for i in range(n):
        p = path_str(i)
        for s in range(k):
            lines.append(p + symbols[s])
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
