import sys, random

# gen.py <testId>  -- prints ONE on-chip-network design instance to stdout.
#
# N nodes sit on a physical ring (the die's wiring channel). Linking node i and
# node j costs cost(i,j) = ring_distance(i,j) = min(|i-j|, N-|i-j|)  (>=1) --
# nearby links are cheap, cross-die links are expensive. A total *link budget*
# L_max caps the sum of costs you may spend building the topology.
#
# Traffic T[i][j] (units routed from i to j per cycle) is mostly a flat
# background (bg=1 everywhere) PLUS, on 7 of the 10 tests, a handful of HOT
# pairs placed near-diametrically opposite on the ring with heavy weight.
# Any topology that only links physically-nearby nodes (a "mesh") forces those
# hot flows through many hops and/or funnels them onto the same few backbone
# edges (a bisection-bandwidth bottleneck) -- the insight is spending scarce
# budget on a few express links that land exactly on the measured hot pairs.

def ring_cost(i, j, N):
    d = abs(i - j)
    return min(d, N - d)

SIZES = {1: 8, 2: 9, 3: 10, 4: 12, 5: 14, 6: 16, 7: 18, 8: 20, 9: 22, 10: 24}
# 7 of the 10 tests plant hot (non-uniform) traffic; 3 stay mild/uniform.
HOT = {2, 3, 4, 5, 6, 7, 8}

def main():
    tid = int(sys.argv[1])
    N = SIZES[tid]
    rng = random.Random(70211 + 1009 * tid)

    T = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                T[i][j] = 1  # flat all-to-all background load

    if tid in HOT:
        H = rng.randint(4, 6)          # number of hot pairs planted
        chosen = set()
        tries = 0
        while len(chosen) < H and tries < 200:
            tries += 1
            i = rng.randrange(N)
            off = N // 2 + rng.randint(-1, 1)
            j = (i + off) % N
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            chosen.add(key)
        weights = {}
        for key in chosen:
            weights[key] = rng.randint(70, 160)
        for (i, j), w in weights.items():
            T[i][j] += w
            T[j][i] += w
        express_slots = max(2, len(chosen) - 2)
    else:
        # mild extra noise, no far concentrated hotspot
        for _ in range(N):
            i = rng.randrange(N); j = rng.randrange(N)
            if i != j:
                T[i][j] += rng.randint(1, 4)
        express_slots = 0

    ring_budget = N  # cost of the N cost-1 ring edges
    if tid in HOT:
        L_max = ring_budget + express_slots * (N // 2)
    else:
        L_max = ring_budget + 2 * N  # generous local-mesh room, no urgent hotspot

    # CAP / STALL_COST: capacity per link before a congestion stall is charged,
    # and the per-unit-over-capacity stall penalty. Scaled to N so the
    # congestion term matters but never swamps the hop-count term.
    CAP = max(6, N)          # link capacity (traffic units) before stalling
    STALL_COST = 4           # penalty per unit of traffic over capacity

    out = [str(N), str(L_max), str(CAP), str(STALL_COST)]
    for row in T:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
