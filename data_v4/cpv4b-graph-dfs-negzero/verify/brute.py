import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    if n == 0:
        print(0)
        return
    v = [0] * (n + 1)
    for i in range(1, n + 1):
        v[i] = int(data[idx]); idx += 1
    children = [[] for _ in range(n + 1)]
    parent = [0] * (n + 1)
    for i in range(2, n + 1):
        p = int(data[idx]); idx += 1
        parent[i] = p
        children[p].append(i)

    # Enumerate EVERY downstream segment (chain from some start node going down,
    # each next node a child of the previous), require >= 1 node, take max sum.
    # Brute force: from every start node, DFS all downward chains, track running sum.
    best = None

    def dfs(u, running):
        nonlocal best
        s = running + v[u]
        if best is None or s > best:
            best = s
        for c in children[u]:
            dfs(c, s)

    # A segment may START at any node, and the chain goes downward from there.
    for start in range(1, n + 1):
        dfs(start, 0)

    print(best)

main()
