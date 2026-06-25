import sys
from collections import deque

def solve(data):
    idx = 0
    m = data[idx]; idx += 1
    n = data[idx]; idx += 1
    masks = []
    for _ in range(n):
        s = data[idx] % m; idx += 1
        L = data[idx]; idx += 1
        msk = 0
        for k in range(L):
            msk |= (1 << ((s + k) % m))
        masks.append(msk)
    full = (1 << m) - 1
    union = 0
    for msk in masks:
        union |= msk
    if union != full:
        return -1
    # BFS over covered-marker bitmask; layer index = number of arcs used.
    dist = {0: 0}
    dq = deque([0])
    while dq:
        cur = dq.popleft()
        if cur == full:
            return dist[cur]
        d = dist[cur]
        for msk in masks:
            nxt = cur | msk
            if nxt not in dist:
                dist[nxt] = d + 1
                dq.append(nxt)
    return dist.get(full, -1)

def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))

if __name__ == "__main__":
    main()
