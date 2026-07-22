import sys, random

SLACK = 12  # extra rack rows beyond the theoretical minimum ceil(log2 N)

# difficulty / trap ladder: (N, L, num_cliques, p_stay)
# num_cliques == 1 (testId 1) => no adjacency structure at all: a pure frequency-skew
# control case. num_cliques >= 2 with high p_stay plants the trap: chime IDs are
# assigned to cliques round-robin (id-1) % num_cliques, so consecutive IDs never share
# a clique, and within-clique co-occurrence dominates the peal -- a canonical
# ID-order / frequency-only construction cannot see this structure at all.
LADDER = [
    (8,   200,   1,  1.00),   # 1: control, freq skew only
    (12,  400,   2,  0.94),   # 2: trap
    (20,  800,   3,  0.94),   # 3: trap
    (32,  1600,  3,  0.95),   # 4: trap
    (48,  2500,  4,  0.95),   # 5: trap
    (70,  4000,  5,  0.96),   # 6: trap
    (100, 6500,  6,  0.96),   # 7: trap
    (150, 10000, 7,  0.97),   # 8: trap
    (220, 18000, 9,  0.97),   # 9: trap, large
    (320, 30000, 11, 0.97),   # 10: trap, largest / perf stress
]


def ceil_log2(n):
    w = 0
    while (1 << w) < n:
        w += 1
    return w


def build_cliques(rng, N, num_cliques):
    clique_of = {}
    members = [[] for _ in range(num_cliques)]
    for i in range(1, N + 1):
        c = (i - 1) % num_cliques
        clique_of[i] = c
        members[c].append(i)
    weights = {}
    for c in range(num_cliques):
        order = members[c][:]
        rng.shuffle(order)
        for rank, sid in enumerate(order, start=1):
            weights[sid] = 0.4 ** (rank - 1)  # steep within-clique skew
    return members, weights


def gen_trace(rng, N, L, num_cliques, members, weights, p_stay):
    cur_c = rng.randrange(num_cliques)
    trace = []
    for _ in range(L):
        mem = members[cur_c]
        wts = [weights[s] for s in mem]
        sid = rng.choices(mem, weights=wts, k=1)[0]
        trace.append(sid)
        if num_cliques > 1 and rng.random() >= p_stay:
            choices = [c for c in range(num_cliques) if c != cur_c]
            cur_c = rng.choice(choices)
    return trace


def main():
    testId = int(sys.argv[1])
    idx = max(1, min(len(LADDER), testId)) - 1
    N, L, num_cliques, p_stay = LADDER[idx]
    rng = random.Random(90210 + 97 * testId)

    Dmax = ceil_log2(N) + SLACK

    members, weights = build_cliques(rng, N, num_cliques)
    trace = gen_trace(rng, N, L, num_cliques, members, weights, p_stay)

    out = [f"{N} {Dmax}", str(L), " ".join(map(str, trace))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
