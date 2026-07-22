import sys, random

# ---------------------------------------------------------------------------
# peloton-courier-drafting : generic weighted city-street graph.
# Internally it is a shared TRUNK corridor (T_0..T_L) plus a private spur
# from every courier's start onto the trunk and a private exit spur to its
# target.  The trunk is the ONLY route, so all couriers share it -- but going
# ASAP their trunk crossings are staggered in TIME (distinct spur phases), so
# the obvious independent-shortest-path plan never drafts.  Node ids are
# permuted so the corridor is not visible from the numbering.
# ---------------------------------------------------------------------------

# testId -> (trunk length L, #core couriers spanning full trunk, #partial couriers)
LADDER = {
    1:  (6,  3, 0),
    2:  (8,  3, 1),
    3:  (10, 4, 1),
    4:  (6,  5, 1),
    5:  (14, 3, 2),
    6:  (12, 4, 2),
    7:  (18, 4, 2),
    8:  (8,  6, 2),
    9:  (22, 3, 3),
    10: (16, 5, 3),
}

TRUNK_W = 10          # base energy of a trunk edge (dominates)
SPUR_W  = 1           # base energy of a spur edge
ALPHA   = 0.30        # leadership streak penalty coefficient
BETA    = 0.15        # drafting energy factor (drafter pays BETA * base)


def main():
    tid = int(sys.argv[1])
    L, kcore, kpart = LADDER[tid]
    rng = random.Random(4000 + tid)

    # logical node ids: trunk nodes are 0..L
    edges = []                      # (u, v, w)
    next_id = L + 1

    def new_node():
        nonlocal next_id
        v = next_id
        next_id += 1
        return v

    for k in range(L):
        edges.append((k, k + 1, TRUNK_W))

    # courier specs: (a, b, spurlen, phase, start, target)
    specs = []
    phase_used = set()

    def add_courier(a, b, phase):
        # spurlen chosen so that (spurlen - a) == phase  and spurlen >= 1
        spurlen = phase + a
        while spurlen < 1:
            spurlen += 1
        ph = spurlen - a
        while ph in phase_used:
            spurlen += 1
            ph = spurlen - a
        phase_used.add(ph)
        # build entry spur:  start -> ... -> trunk[a]   (spurlen edges)
        prev = new_node()
        start = prev
        for _ in range(spurlen - 1):
            nxt = new_node()
            edges.append((prev, nxt, SPUR_W))
            prev = nxt
        edges.append((prev, a, SPUR_W))          # last spur edge onto the trunk
        # exit spur: trunk[b] -> target   (1 edge)
        tgt = new_node()
        edges.append((b, tgt, SPUR_W))
        specs.append((a, b, spurlen, ph, start, tgt))

    for j in range(kcore):
        add_courier(0, L, j + 1)                 # phases 1..kcore, distinct

    nextphase = kcore + 1
    for _ in range(kpart):
        a = rng.randint(1, max(1, L - 3))
        b = rng.randint(a + 2, L)
        add_courier(a, b, nextphase)
        nextphase += 1

    # global convoy offset used only to size the deadlines so the intended
    # synchronized (strong/greedy) schedule is guaranteed feasible.
    OFF = max([0] + [ph for (_, _, _, ph, _, _) in specs])

    couriers = []                                # (start, target, deadline)
    for (a, b, spurlen, ph, start, tgt) in specs:
        strong_arrival = OFF + b + 1             # +1 exit spur edge
        deadline = strong_arrival + 2            # small slack
        couriers.append((start, tgt, deadline))

    # ---- permute node ids so the trunk is hidden ----
    N = next_id
    perm = list(range(N))
    rng.shuffle(perm)                            # perm[old] = new id

    def relabel(x):
        return perm[x]

    edges = [(relabel(u), relabel(v), w) for (u, v, w) in edges]
    couriers = [(relabel(s), relabel(g), d) for (s, g, d) in couriers]

    rng.shuffle(edges)
    order = list(range(len(couriers)))
    rng.shuffle(order)
    couriers = [couriers[i] for i in order]

    out = []
    out.append("%d %d %d" % (N, len(edges), len(couriers)))
    out.append("%.4f %.4f" % (ALPHA, BETA))
    for (u, v, w) in edges:
        out.append("%d %d %d" % (u, v, w))
    for (s, g, d) in couriers:
        out.append("%d %d %d" % (s, g, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
