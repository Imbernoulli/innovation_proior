# TIER: strong
# Continuation-value allocation: a rebate is worth its cascade CONTINUATION, not the
# recipient's own adoption. Repeatedly activate the household whose *current* activation
# gap -- after crediting already-adopted upstream neighbors -- is cheapest, paying exactly
# that gap. Entering a corridor costs one interior's self-price, but every downstream
# step then costs only the near-threshold top-up, so the budget fragments into long chains.
import sys, heapq

def main():
    it = iter(sys.stdin.buffer.read().split())
    n = int(next(it)); m = int(next(it)); B = int(next(it)); W = int(next(it))
    theta = [0]*n; r = [0]*n
    for i in range(n):
        theta[i] = int(next(it)); r[i] = int(next(it))
    succ = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)); v = int(next(it)); succ[u].append(v)

    indeg = [0]*n
    adopted = bytearray(n)
    x = [0]*n; rem = B

    def gap(i):
        need = theta[i] - W*indeg[i]
        if need <= 0:
            return 0
        return -(-need // r[i])           # ceil(need / r_i)

    h = [(gap(i), i) for i in range(n)]
    heapq.heapify(h)
    while h:
        c, i = heapq.heappop(h)
        if adopted[i]:
            continue
        cur = gap(i)
        if cur != c:
            heapq.heappush(h, (cur, i)); continue
        if cur > rem:
            break                          # nothing cheaper is affordable
        x[i] = cur; rem -= cur; adopted[i] = 1
        for v in succ[i]:
            if not adopted[v]:
                indeg[v] += 1
                heapq.heappush(h, (gap(v), v))

    sys.stdout.write(" ".join(map(str, x)) + "\n")

main()
