import sys
from collections import deque

# Independent brute force: BFS over states 0..n where an edge v -> v - p
# exists for every perfect k-th power p <= v. The shortest path from n to 0
# (number of edges) is the minimum number of perfect k-th powers summing to n.
# Completely different method from the forward DP in sol.cpp.

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    k = int(data[0]); n = int(data[1])
    if n == 0:
        print(0); return

    powers = []
    b = 1
    while b ** k <= n:
        powers.append(b ** k)
        b += 1

    # BFS from n down to 0.
    dist = [-1] * (n + 1)
    dist[n] = 0
    q = deque([n])
    while q:
        v = q.popleft()
        if v == 0:
            break
        for p in powers:
            if p > v:
                break
            w = v - p
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                q.append(w)
    print(dist[0])

main()
