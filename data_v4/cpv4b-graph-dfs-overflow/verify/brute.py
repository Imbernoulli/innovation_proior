import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    parent = [0] * (n + 1)
    children = [[] for _ in range(n + 1)]
    for v in range(2, n + 1):
        p = int(data[idx]); idx += 1
        parent[v] = p
        children[p].append(v)

    if n == 0:
        print(0)
        return

    # Depth of each node by walking up to the root (root = node 1, depth 0).
    # Independent and brute: for each node v, count its subtree by checking every
    # node u and testing whether v is an ancestor of u (including u == v).
    def depth_of(v):
        d = 0
        while v != 1:
            v = parent[v]
            d += 1
        return d

    def is_ancestor(a, u):
        # is a an ancestor of u, inclusive (a == u counts)?
        while True:
            if u == a:
                return True
            if u == 1:
                return False
            u = parent[u]

    depth = [0] * (n + 1)
    for v in range(1, n + 1):
        depth[v] = depth_of(v)

    total = 0
    for v in range(1, n + 1):
        size = 0
        for u in range(1, n + 1):
            if is_ancestor(v, u):
                size += 1
        total += size * depth[v]

    print(total)

main()
