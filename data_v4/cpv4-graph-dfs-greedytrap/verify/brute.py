import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    p = []
    for _ in range(n):
        p.append(int(data[idx])); idx += 1
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        adj[u].append(v)

    # Brute force: EXHAUSTIVELY enumerate every directed path (the graph is a
    # DAG, so every path is finite and acyclic). For each path we compute the
    # total prestige of its nodes and keep the global maximum. No DP, no memo,
    # no cleverness -- this is obviously correct and uses a different method
    # from the memoized solution.
    best = -10**18

    def walk(u, acc):
        nonlocal best
        acc += p[u]            # we are reading u right now
        if acc > best:
            best = acc          # stopping here is a valid path
        for v in adj[u]:
            walk(v, acc)        # extend to neighbour v

    sys.setrecursionlimit(1000000)
    for s in range(n):
        walk(s, 0)             # a path may start at any node

    print(best)

main()
