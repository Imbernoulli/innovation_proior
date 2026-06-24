import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1

    adj = [set() for _ in range(n + 1)]
    edges = []
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        edges.append((u, v))

    def connected(adj_local, a, b):
        # plain BFS reachability on current graph
        if a == b:
            return True
        seen = {a}
        stack = [a]
        while stack:
            x = stack.pop()
            for y in adj_local[x]:
                if y not in seen:
                    if y == b:
                        return True
                    seen.add(y)
                    stack.append(y)
        return False

    redundant = 0
    prefixRedundantSum = 0
    for (u, v) in edges:
        # is this cable redundant given edges added so far?
        if connected(adj, u, v):
            redundant += 1
        # add the cable to the graph
        adj[u].add(v)
        adj[v].add(u)
        prefixRedundantSum += redundant

    # count unordered same-component pairs from final graph via component sizes
    seen = [False] * (n + 1)
    samePairs = 0
    for s in range(1, n + 1):
        if not seen[s]:
            comp = 0
            stack = [s]
            seen[s] = True
            while stack:
                x = stack.pop()
                comp += 1
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
            samePairs += comp * (comp - 1) // 2

    print(redundant, samePairs, prefixRedundantSum)

main()
