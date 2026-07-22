#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for latent sparse-graph diffusion identification.

- Reads ONLY the test id `t` from <in>'s header (the rest of <in> is ignored --
  the hidden network is regenerated here, byte-identically to gen.py, from `t`
  alone).  The hidden network, the training source set and the HELD-OUT source
  set all live only inside this function; nothing is imported from gen.py.
- The held-out grading experiments are impulses at N_TEST_SRC=4 junctions that
  were NOT among the 5 training sources, rolled forward for T_TEST steps under
  the TRUE hidden operator A, then perturbed by a small irreducible
  observation-noise floor (never given to the solver).
- Parses the participant's submitted graph -- a weighted edge list -- builds
  the implied symmetric leaky-diffusion operator A_hat, and rolls A_hat
  forward on the SAME held-out impulses.  Score = held-out rollout MSE with a
  small edge-count parsimony term (minimisation):
      F = heldout_MSE(A_hat)   * (1 + LAMBDA * m_edges)
      B = heldout_MSE(identity)* (1 + LAMBDA * 0)     # baseline: no diffusion
      Ratio = min(1000, 100*B/F) / 1000
  Submitting zero edges (no diffusion) reproduces B exactly (~0.1).  A graph
  that recovers the true sparse operator drives the held-out MSE far below the
  no-diffusion baseline; noise plus estimation error keep even a very good
  operator well under the cap, leaving headroom.
"""
import sys, random, math

N = 30
WMIN, WMAX = 0.05, 0.13
DEGCAP = 0.55
T_TRAIN = 14
N_TRAIN_SRC = 5
T_TEST = 12
N_TEST_SRC = 4
HELD_NOISE = 0.004
LAMBDA = 0.02
ROWCAP = 1.0 + 1e-9
MAX_EDGES = N * (N - 1) // 2          # 435
MAX_OUT_BYTES = 300000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden network (IDENTICAL construction to gen.py) ----------
def build_graph(t):
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


IDENTITY = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]


def heldout(t):
    edges = build_graph(t)
    A = edges_to_A(edges)
    train_src = pick_sources(t, N_TRAIN_SRC, salt=1)
    test_src = pick_sources(t, N_TEST_SRC, exclude=set(train_src), salt=2)
    rng = random.Random(88000 + t * 17)
    targets = []
    for s in test_src:
        true_traj = simulate(A, s, T_TEST)
        noisy = [row[:] for row in true_traj]
        for tt in range(1, T_TEST + 1):
            for j in range(N):
                noisy[tt][j] += rng.gauss(0.0, HELD_NOISE)
        targets.append((s, noisy))
    return targets


# ---------- participant output parsing ----------
def parse_edges(raw, n=N):
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    toks = text.split()
    if not toks:
        fail("empty output")
    try:
        m = int(toks[0])
    except Exception:
        fail("bad edge count header")
    if m < 0 or m > MAX_EDGES:
        fail("edge count out of range")
    need = 1 + 3 * m
    if len(toks) != need:
        fail("token count mismatch for declared edge count")
    edges = {}
    row = [0.0] * n
    pos = 1
    for _ in range(m):
        si, sj, sw = toks[pos], toks[pos + 1], toks[pos + 2]
        pos += 3
        try:
            i = int(si)
            j = int(sj)
            w = float(sw)
        except Exception:
            fail("malformed edge token")
        if w != w or w in (float("inf"), float("-inf")):
            fail("non-finite weight")
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            fail("node index out of range / self-loop")
        if not (0.0 < w <= 1.0):
            fail("edge weight out of range (0,1]")
        key = (min(i, j), max(i, j))
        if key in edges:
            fail("duplicate edge")
        edges[key] = w
        row[i] += w
        row[j] += w
    for i in range(n):
        if row[i] > ROWCAP:
            fail("node %d incident weight sum exceeds 1.0 (infeasible diffusion)" % i)
    return edges, m


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")

    edges, m = parse_edges(raw)
    Ahat = edges_to_A(edges)

    targets = heldout(t)
    F_num = 0.0
    B_num = 0.0
    F_den = 0
    for s, noisy in targets:
        pred = simulate(Ahat, s, T_TEST)
        base = simulate(IDENTITY, s, T_TEST)
        for tt in range(1, T_TEST + 1):
            for j in range(N):
                F_num += (pred[tt][j] - noisy[tt][j]) ** 2
                B_num += (base[tt][j] - noisy[tt][j]) ** 2
                F_den += 1

    F_mse = F_num / F_den
    B_mse = B_num / F_den
    F = F_mse * (1.0 + LAMBDA * m)
    B = B_mse * (1.0 + LAMBDA * 0)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f edges=%d  Ratio: %.6f"
          % (F_mse, B_mse, m, sc / 1000.0))


if __name__ == "__main__":
    main()
