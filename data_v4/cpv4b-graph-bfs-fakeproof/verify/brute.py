import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        adj[u].append(v)
        adj[v].append(u)

    # BFS from node 1.
    d = [-1] * (n + 1)
    d[1] = 0
    q = deque([1])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if d[y] == -1:
                d[y] = d[x] + 1
                q.append(y)

    # Brute force: sum over all unordered pairs {u,v}, u<v, of popcount(d[u] XOR d[v]).
    total = 0
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            x = d[u] ^ d[v]
            total += bin(x).count("1")
    print(total)

if __name__ == "__main__":
    main()
