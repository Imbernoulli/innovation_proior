import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    p = [int(data[idx + i]) for i in range(n)]; idx += n
    K = int(data[idx]); idx += 1

    # Independent brute force: explicitly build the sorted set of DISTINCT pulse times by
    # a k-way merge using a heap, popping until we have emitted the K-th distinct time.
    # Each machine i yields the infinite sequence p[i], 2*p[i], 3*p[i], ...
    import heapq
    heap = []
    for i in range(n):
        heapq.heappush(heap, (p[i], i))  # (current time, machine index)

    emitted = 0
    last = None
    ans = None
    while heap:
        t, i = heapq.heappop(heap)
        # advance this machine to its next pulse
        # recover the multiple count: t // p[i]
        k = t // p[i]
        heapq.heappush(heap, ((k + 1) * p[i], i))
        if t != last:        # new DISTINCT time
            last = t
            emitted += 1
            if emitted == K:
                ans = t
                break
    print(ans)

main()
