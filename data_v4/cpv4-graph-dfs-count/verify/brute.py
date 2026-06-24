import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    edges = []
    for _ in range(m):
        a = int(data[idx]); b = int(data[idx+1]); idx += 2
        edges.append((a, b))

    # An edge is a BRIDGE iff removing it increases the number of connected
    # components. Equivalently (for THIS edge instance), its two endpoints are
    # no longer connected using all OTHER edges.
    # An edge lies on at least one cycle  <=>  it is NOT a bridge.
    # (Self-loops: endpoints already equal -> always still "connected" via the
    #  trivial 0-length path -> never a bridge -> always counts.)

    def connected_without(skip_index, x, y):
        if x == y:
            return True
        # build adjacency excluding the single edge instance skip_index
        g = [[] for _ in range(n + 1)]
        for i, (a, b) in enumerate(edges):
            if i == skip_index:
                continue
            g[a].append(b)
            g[b].append(a)
        seen = [False] * (n + 1)
        stack = [x]
        seen[x] = True
        while stack:
            u = stack.pop()
            if u == y:
                return True
            for w in g[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        return seen[y]

    non_bridges = 0
    for i, (a, b) in enumerate(edges):
        if connected_without(i, a, b):
            non_bridges += 1  # endpoints still connected without this edge => not a bridge => on a cycle
    print(non_bridges)

main()
