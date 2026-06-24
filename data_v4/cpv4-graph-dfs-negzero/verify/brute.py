import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    v = []
    for _ in range(n):
        v.append(int(data[idx])); idx += 1
    adj = [[] for _ in range(n)]
    for _ in range(m):
        a = int(data[idx]); idx += 1
        b = int(data[idx]); idx += 1
        adj[a].append(b)

    # Brute: enumerate every directed walk starting at chamber 0 following edges.
    # Because every edge goes a < b (strictly deeper), every walk is a simple path
    # of finite length; we DFS over all such walks and track the running sum at every
    # prefix (the explorer may STOP at any chamber). Answer = best running sum reachable
    # from a walk that starts at 0. We always include v[0] (we start there).
    best = [None]

    def walk(u, acc):
        acc += v[u]
        if best[0] is None or acc > best[0]:
            best[0] = acc
        for c in adj[u]:
            walk(c, acc)

    walk(0, 0)
    print(best[0])

main()
