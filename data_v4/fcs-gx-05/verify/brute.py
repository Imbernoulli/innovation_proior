#!/usr/bin/env python3
# Independent oracle: decide feasibility via bipartite maximum matching
# (Kuhn's augmenting-path algorithm) between requests and time slots.
# Request i is adjacent to every slot s with l[i] <= s <= r[i].
# All requests can be served iff the maximum matching saturates all requests.
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    T = int(next(it))
    l = [0] * n
    r = [0] * n
    for i in range(n):
        l[i] = int(next(it))
        r[i] = int(next(it))

    # match_slot[s] = request currently assigned to slot s, or -1.
    match_slot = [-1] * (T + 2)

    def try_kuhn(u, visited):
        # Attempt to find an augmenting path for request u.
        for s in range(l[u], r[u] + 1):
            if not visited[s]:
                visited[s] = True
                if match_slot[s] == -1 or try_kuhn(match_slot[s], visited):
                    match_slot[s] = u
                    return True
        return False

    matched = 0
    for u in range(n):
        visited = [False] * (T + 2)
        if try_kuhn(u, visited):
            matched += 1

    if matched == n:
        # Reconstruct assignment per request.
        assign = [-1] * n
        for s in range(1, T + 1):
            if match_slot[s] != -1:
                assign[match_slot[s]] = s
        print("YES")
        print(" ".join(str(x) for x in assign))
    else:
        print("NO")

if __name__ == "__main__":
    main()
