# TIER: strong
"""Portfolio contraction optimizer.  Runs several deterministic policies
(min-cost greedy, min-resulting-size greedy) plus a batch of seeded
randomized-greedy restarts, simulates the exact cost of each full schedule,
and emits the cheapest one found.  No proven-optimal scheme -- the true
optimum of a general (cyclic) tensor network is NP-hard and unknown."""
import sys
import random


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


def prod(dims, S):
    c = 1
    for i in S:
        c *= dims[i]
    return c


def contract(m, dims, tensors, output, chooser):
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
        a, b, newset, cost = chooser(live, dims, count, output)
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


def evaluate(live, dims, count, output, a, b):
    SA, SB = live[a], live[b]
    U = SA | SB
    cost = prod(dims, U)
    newset = set()
    for i in U:
        others = count[i] - (1 if i in SA else 0) - (1 if i in SB else 0)
        if output[i] or others > 0:
            newset.add(i)
    return newset, cost


def make_chooser(mode, rng, rank="cost"):
    def chooser(live, dims, count, output):
        ids = sorted(live)
        cands = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                shared = 1 if (live[a] & live[b]) else 0
                newset, cost = evaluate(live, dims, count, output, a, b)
                if mode == "size" or rank == "size":
                    key = (0 if shared else 1, prod(dims, newset), cost, a, b)
                else:  # cost
                    key = (0 if shared else 1, cost, prod(dims, newset), a, b)
                cands.append((key, a, b, newset, cost))
        cands.sort(key=lambda t: t[0])
        if mode == "rand":
            top = cands[:max(1, min(4, len(cands)))]
            pick = rng.choice(top)
        else:
            pick = cands[0]
        _, a, b, newset, cost = pick
        return a, b, newset, cost
    return chooser


def main():
    m, dims, tensors, output = read_instance()
    if m < 2:
        sys.stdout.write("")
        return

    best_merges = None
    best_cost = None

    def consider(merges, cost):
        nonlocal best_merges, best_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_merges = merges

    # deterministic policies
    consider(*contract(m, dims, tensors, output, make_chooser("cost", None)))
    consider(*contract(m, dims, tensors, output, make_chooser("size", None)))

    # seeded randomized restarts (cost- and size-ranked)
    for s in range(40):
        rng = random.Random(1234567 + s * 6971)
        consider(*contract(m, dims, tensors, output, make_chooser("rand", rng, "cost")))
        rng2 = random.Random(7654321 + s * 4099)
        consider(*contract(m, dims, tensors, output, make_chooser("rand", rng2, "size")))

    sys.stdout.write("\n".join("%d %d" % (a, b) for a, b in best_merges) + "\n")


if __name__ == "__main__":
    main()
