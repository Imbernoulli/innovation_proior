# TIER: greedy
"""The obvious textbook approach: sequential FIRST-FIT vertex coloring in raw
input order, always taking the smallest still-available color index.
It is cost-oblivious (never looks at the cost array) and structure-oblivious
(never notices wheel/parity shape), and never attempts an obstruction
certificate -- it has no way to explain why it needed the colors it used."""
import sys


def main():
    data = sys.stdin.read().split()
    ti = 0
    n = int(data[ti]); ti += 1
    m = int(data[ti]); ti += 1
    K = int(data[ti]); ti += 1
    ti += K  # skip costs -- greedy does not use them
    adj = {v: set() for v in range(1, n + 1)}
    for _ in range(m):
        a, b, c = int(data[ti]), int(data[ti + 1]), int(data[ti + 2]); ti += 3
        for (u, v) in ((a, b), (b, c), (a, c)):
            adj[u].add(v)
            adj[v].add(u)

    color = [0] * (n + 1)
    for v in range(1, n + 1):
        used = {color[u] for u in adj[v] if color[u] != 0}
        chosen = 1
        for c in range(1, K + 1):
            if c not in used:
                chosen = c
                break
        color[v] = chosen

    out = []
    out.append(" ".join(str(color[v]) for v in range(1, n + 1)))
    out.append("0")  # never certifies -- it has no obstruction-detection logic
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
