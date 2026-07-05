# TIER: greedy
"""Two-policy greedy: build both a min-cost greedy schedule and a
min-resulting-size greedy schedule, simulate the exact cost of each, and emit
the cheaper one.  Never worse than the min-cost baseline and often better."""
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    k = int(next(it))
    dims = [int(next(it)) for _ in range(k)]
    tensors = []
    for _ in range(m):
        deg = int(next(it))
        tensors.append([int(next(it)) for _ in range(deg)])
    occ = [0] * k
    for t in tensors:
        for i in t:
            occ[i] += 1
    output = [c == 1 for c in occ]
    return m, dims, tensors, output


def contract(m, dims, tensors, output, mode):
    live = {}
    count = {}
    for tid in range(m):
        s = set(tensors[tid])
        live[tid] = s
        for i in s:
            count[i] = count.get(i, 0) + 1
    nid = m
    merges = []
    total = 0
    while len(live) > 1:
        ids = sorted(live)
        best = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                SA, SB = live[a], live[b]
                U = SA | SB
                cost = 1
                for i in U:
                    cost *= dims[i]
                newset = set()
                for i in U:
                    others = count[i] - (1 if i in SA else 0) - (1 if i in SB else 0)
                    if output[i] or others > 0:
                        newset.add(i)
                rsize = 1
                for i in newset:
                    rsize *= dims[i]
                shared = 0 if (SA & SB) else 1
                if mode == "size":
                    key = (shared, rsize, cost, a, b)
                else:
                    key = (shared, cost, rsize, a, b)
                if best is None or key < best[0]:
                    best = (key, a, b, newset, cost)
        _, a, b, newset, cost = best
        SA = live.pop(a)
        SB = live.pop(b)
        for i in SA:
            count[i] -= 1
        for i in SB:
            count[i] -= 1
        for i in newset:
            count[i] = count.get(i, 0) + 1
        live[nid] = newset
        merges.append((a, b))
        total += cost
        nid += 1
    return merges, total


def main():
    m, dims, tensors, output = read_instance()
    if m < 2:
        sys.stdout.write("")
        return
    mc, tc = contract(m, dims, tensors, output, "cost")
    ms, ts = contract(m, dims, tensors, output, "size")
    merges = mc if tc <= ts else ms
    sys.stdout.write("\n".join("%d %d" % (a, b) for a, b in merges) + "\n")


if __name__ == "__main__":
    main()
