import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1

    # Independent brute force: maintain an explicit adjacency-based component
    # labeling. After each edge, recompute connected components from scratch
    # via BFS/flood fill over all edges seen so far, then count connected pairs
    # as sum over components of C(size, 2). This is O(m * (n + edges)) -- only
    # valid for tiny cases, which is exactly what we use it for.
    edges = []
    out = []
    grand = 0
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        edges.append((u, v))

        # build adjacency over current edge set
        adj = [[] for _ in range(n + 1)]
        for (a, b) in edges:
            adj[a].append(b)
            adj[b].append(a)

        seen = [False] * (n + 1)
        pairs = 0
        for start in range(1, n + 1):
            if seen[start]:
                continue
            # BFS this component
            stack = [start]
            seen[start] = True
            comp_size = 0
            while stack:
                x = stack.pop()
                comp_size += 1
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
            pairs += comp_size * (comp_size - 1) // 2

        grand += pairs
        out.append(str(pairs))

    out.append(str(grand))
    sys.stdout.write("\n".join(out) + "\n")

main()
