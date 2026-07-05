import sys, random

# gen.py <testId>  -- prints ONE qubit-routing instance to stdout.
#
# We model an IBM-style superconducting device: physical qubits sit on a 2D
# lattice ("coupling map"); a two-qubit gate can only be executed between two
# PHYSICALLY ADJACENT qubits.  A fixed schedule of logical two-qubit
# interactions must be executed IN ORDER.  Because interacting logical qubits
# are usually not adjacent, the compiler inserts SWAP gates (each swap exchanges
# the logical contents of two adjacent physical qubits) to route them together.
#
# The task (see statement) is to choose an initial placement of logical qubits
# onto physical qubits and a sequence of SWAPs that makes every scheduled
# interaction executable, MINIMIZING the total number of inserted SWAPs.  Qubit
# routing / SWAP minimization is NP-hard; the true optimum is unknown, so the
# problem stays genuinely open-ended.
#
# Instances have LOCALITY: logical qubits are grouped into interaction clusters,
# so a good initial placement (clusters -> nearby lattice regions) and a
# look-ahead router materially beat the naive identity-placement / greedy
# shortest-path baseline.  Difficulty grows with testId (larger lattice, more
# gates).

# (rows, cols, n_gates)   physical qubits = rows*cols == logical qubits
SPECS = {
    1:  (3, 3, 12),
    2:  (3, 4, 16),
    3:  (4, 4, 22),
    4:  (4, 4, 30),
    5:  (4, 5, 34),
    6:  (4, 5, 44),
    7:  (5, 5, 46),
    8:  (5, 5, 58),
    9:  (5, 6, 60),
    10: (5, 6, 74),
}


def grid_edges(rows, cols):
    E = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                E.append((u, r * cols + (c + 1)))
            if r + 1 < rows:
                E.append((u, (r + 1) * cols + c))
    return E


def main():
    tid = int(sys.argv[1])
    rows, cols, M = SPECS[tid]
    P = rows * cols
    L = P
    rng = random.Random(1_000_003 * tid + 7)

    edges = grid_edges(rows, cols)

    # cluster logical qubits so the schedule has exploitable locality
    n_cl = max(2, int(round(L ** 0.5)))
    cluster = [rng.randrange(n_cl) for _ in range(L)]
    by_cl = {}
    for q in range(L):
        by_cl.setdefault(cluster[q], []).append(q)

    gates = []
    for _ in range(M):
        # 65% intra-cluster interaction, else global random pair
        if rng.random() < 0.65:
            cl = rng.randrange(n_cl)
            pool = by_cl[cl]
            if len(pool) >= 2:
                a, b = rng.sample(pool, 2)
            else:
                a, b = rng.sample(range(L), 2)
        else:
            a, b = rng.sample(range(L), 2)
        gates.append((a, b))

    out = ["%d %d %d %d" % (P, len(edges), M, L)]
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    for (a, b) in gates:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
