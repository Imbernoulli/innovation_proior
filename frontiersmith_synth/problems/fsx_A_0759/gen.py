#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN observation set to stdout.

Latent sparse-graph diffusion identification.  A hidden pipe network connects
n=30 junctions; pipe (i,j) has a hidden nonnegative conductance w_ij.  Heat
obeys the symmetric leaky-diffusion recursion (mass-conserving, no environment
leak):

    x[t+1]_i = x[t]_i + sum_{j in N(i)} w_ij * (x[t]_j - x[t]_i)

equivalently x[t+1] = A x[t] with A_ii = 1 - sum_j w_ij, A_ij = A_ji = w_ij on
edges, 0 elsewhere.  Each testId fixes a DIFFERENT hidden sparse network
(average degree ~2.5) with edge conductances drawn independently -- the graph
lives ONLY in this generator's / the checker's private RNG stream and is
NEVER printed.

The solver observes IMPULSE RESPONSES: a technician injects one unit of heat
at a single source junction (all others at 0) and logs the resulting
temperature of EVERY junction for T_TRAIN further steps, with small sensor
noise.  This is done at N_TRAIN_SRC=5 source junctions (fixed per testId).
The held-out grading experiments -- impulses at OTHER junctions, run for a
longer horizon -- are regenerated only inside the checker and never appear
here.

STDOUT prints ONLY: a header, the list of the 5 source ids used, then for
each source (in that order) T_TRAIN+1 rows of n noisy temperatures.  No
hidden conductance, no adjacency, no RNG seed is ever printed.
"""
import sys, random, math

N = 30
WMIN, WMAX = 0.05, 0.13
DEGCAP = 0.55
T_TRAIN = 14
N_TRAIN_SRC = 5
NOISE_SIGMA = 0.0045


def build_graph(t):
    """Hidden sparse pipe network for this test id. Lives in gen AND checker;
    never printed. Backbone = random spanning permutation-chain (keeps the
    network connected and inherently path-heavy / slow-mixing, like a real
    pipe grid), plus extra random pipes up to average degree ~2.5."""
    rng = random.Random(759000 + t * 104729)
    perm = list(range(N))
    rng.shuffle(perm)
    edges = {}
    for i in range(1, N):
        a, b = perm[i - 1], perm[i]
        edges[(min(a, b), max(a, b))] = rng.uniform(WMIN, WMAX)
    target_edges = int(round(N * 2.5 / 2))
    tries = 0
    while len(edges) < target_edges and tries < 5000:
        tries += 1
        a, b = rng.randrange(N), rng.randrange(N)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edges:
            continue
        edges[key] = rng.uniform(WMIN, WMAX)
    deg = [0.0] * N
    for (a, b), w in edges.items():
        deg[a] += w
        deg[b] += w
    mx = max(deg)
    if mx > DEGCAP:
        scale = DEGCAP / mx
        edges = {k: v * scale for k, v in edges.items()}
    return edges


def edges_to_A(edges, n=N):
    A = [[0.0] * n for _ in range(n)]
    deg = [0.0] * n
    for (a, b), w in edges.items():
        A[a][b] = w
        A[b][a] = w
        deg[a] += w
        deg[b] += w
    for i in range(n):
        A[i][i] = 1.0 - deg[i]
    return A


def matvec(A, x, n=N):
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def pick_sources(t, k, exclude=(), salt=0):
    rng = random.Random(759000 + t * 104729 + 999983 * salt)
    pool = [i for i in range(N) if i not in exclude]
    rng.shuffle(pool)
    return sorted(pool[:k])


def simulate(A, source, T, n=N):
    x = [0.0] * n
    x[source] = 1.0
    traj = [x[:]]
    for _ in range(T):
        x = matvec(A, x, n)
        traj.append(x[:])
    return traj


def train_data(t):
    edges = build_graph(t)
    A = edges_to_A(edges)
    train_src = pick_sources(t, N_TRAIN_SRC, salt=1)
    rng = random.Random(20000 + t * 31)
    blocks = []
    for s in train_src:
        traj = simulate(A, s, T_TRAIN)
        noisy = [row[:] for row in traj]
        for tt in range(1, T_TRAIN + 1):
            for j in range(N):
                noisy[tt][j] += rng.gauss(0.0, NOISE_SIGMA)
        blocks.append((s, noisy))
    return train_src, blocks


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train_src, blocks = train_data(t)
    out = []
    out.append("%d %d %d %d" % (t, N, N_TRAIN_SRC, T_TRAIN))
    out.append(" ".join(str(s) for s in train_src))
    for s, noisy in blocks:
        for row in noisy:
            out.append(" ".join("%.6f" % v for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
