#!/usr/bin/env python3
"""
gen.py -- generator for fsx_B_0884 (nonlinear-propagation-flop-schedule).

Emits one instance of a saturating community-diffusion fixed-point problem:

    N agents, each in one of K communities (block id c_i).
    W[a][b]        per-community-pair coupling weight (integer, >=0)
    CAP            saturation cap (integer, >=1)

    x_i^(0) = 0
    x_i^(t+1) = min(CAP, beta[c_i] + sum_{j != i} W[c_i][c_j] * x_j^(t))

beta[0..K-1] (the per-community bias / injection) is deliberately NOT printed
here: it is an unknown that the checker substitutes at several different
values -- the submitted circuit must be correct for EVERY beta in [1, CAP]^K,
not just one. See counter.py for how beta enters (as K implicit input nodes
of the submitted SLP) and statement.md for the contract. This closes off
"just hardcode the numeric answer as a `const`" (beta is exactly the piece of
data that makes hardcoding fail).

This is a mean-field / stochastic-block-model diffusion: the coupling between
two INDIVIDUAL agents i,j depends only on their communities (c_i, c_j), so it
is completely determined by the K x K table W plus the community sizes.

Because the initial condition x^(0)=0 is community-symmetric and the update
rule only depends on community identity, ALL agents in the same community
have IDENTICAL trajectories forever (a graph-automorphism argument) -- so the
true state space is only K-dimensional, not N-dimensional.  The dynamics are
also globally monotone (all weights/biases >= 0) and bounded above by CAP, so
from x^(0)=0 the sequence is coordinate-wise non-decreasing and integer
valued -> it reaches an exact fixed point after at most K*CAP rounds, for
ANY beta in [0, CAP]^K (see counter.py's docstring for the proof; the bound
is beta-independent, which is exactly what lets a fixed circuit -- built
without knowing beta's value -- unroll "enough" rounds safely).

Test-parameter table (testId 1..10): (N, K, CAP) chosen so that N/K grows from
1.0 (no compression possible -- sanity case) up to ~3.0 (a handful of "trap"
cases where naive/per-agent simulation costs asymptotically N/K and N^2/K^2
times more than the community-collapsed circuit).

Determinism: everything is seeded from testId only (random.Random(seed)).
"""
import sys, random


def build_block_graph(K, rng, ring_w_choices, extra_frac, self_choices):
    """Return W (K x K list of lists): a sparse community coupling table that
    is DELIBERATELY ACYCLIC -- community a may only be influenced by
    communities with a SMALLER index (a chain a -> a-1 plus a few extra
    long-range "b < a" links). No self-loops.

    This acyclicity is load-bearing, not cosmetic: with any directed cycle
    (a self-loop included) whose weights/multiplicities are all >= 1, the
    dynamics amplify without bound every round until every community hits
    CAP for ANY beta >= 1 -- i.e. the fixed point would stop depending on
    beta at all, and a solver could "hardcode CAP" for zero cost. Keeping
    the community graph a DAG guarantees community 0 (no incoming edges)
    always equals beta_0 exactly, and downstream communities are genuine
    (bounded, non-constant) linear-then-clamped combinations of upstream
    betas -- so the checker's beta substitution actually has teeth.
    `self_choices` is accepted for API stability but no longer used."""
    W = [[0] * K for _ in range(K)]
    if K >= 2:
        for a in range(1, K):
            W[a][a - 1] = rng.choice(ring_w_choices)
        n_extra = max(0, int(round(extra_frac * K)))
        tries = 0
        added = 0
        while added < n_extra and tries < 20 * K + 20:
            tries += 1
            a = rng.randrange(1, K)
            b = rng.randrange(0, a)          # b < a only -> stays acyclic
            if W[a][b] != 0:
                continue
            W[a][b] = rng.choice(ring_w_choices)
            added += 1
    return W


def mult(a, b, n):
    return n[b] if b != a else max(0, n[a] - 1)


# (N, K, CAP, self_choices, ring_w_choices, extra_frac, skew)
PARAMS = [
    (6,  6,  6,  (0, 1, 2),    (1, 2),    0.30, False),   # 1: N/K=1.00 sanity (no compression)
    (10, 7,  7,  (0, 1, 2),    (1, 2),    0.30, False),   # 2: N/K=1.43
    (16, 8,  8,  (0, 1, 2),    (1, 2),    0.35, False),   # 3: N/K=2.00
    (20, 8,  9,  (0, 1, 2, 3), (1, 2),    0.35, False),   # 4: N/K=2.50
    (24, 8,  10, (0, 1, 2, 3), (1, 2),    0.35, False),   # 5: N/K=3.00
    (28, 10, 8,  (0, 1, 2),    (1, 2),    0.35, False),   # 6: N/K=2.80  trap
    (33, 11, 9,  (0, 1, 2, 3), (1, 2),    0.35, False),   # 7: N/K=3.00  trap
    (36, 12, 10, (0, 1, 2, 3), (1, 2),    0.30, True),    # 8: N/K=3.00  trap (mild skew)
    (40, 13, 11, (0, 1, 2, 3), (1, 2, 3), 0.40, False),   # 9: N/K=3.08  trap
    (45, 15, 12, (0, 1, 2, 3), (1, 2, 3), 0.40, False),   # 10: N/K=3.00 trap (largest)
]


def block_sizes(N, K, rng, skew):
    if skew and K >= 3:
        # one moderately oversized community (~2.5x the uniform share),
        # the rest split the remainder evenly -- a "regime change" the
        # simple N/K-uniform reasoning of `greedy` does not model, but
        # mild enough not to blow past the strong-solution headroom cap.
        big = min(N - (K - 1), int(round(2.5 * N / K)))
        rest = N - big
        if big < N // K + 1 or rest < K - 1:
            skew = False
        else:
            base = rest // (K - 1)
            rem = rest - base * (K - 1)
            sizes = [big] + [base + (1 if a < rem else 0) for a in range(K - 1)]
            rng.shuffle(sizes)
            return sizes
    base = N // K
    rem = N - base * K
    sizes = [base + (1 if a < rem else 0) for a in range(K)]
    rng.shuffle(sizes)
    return sizes


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    assert 1 <= tid <= 10
    N, K, CAP, self_choices, ring_w_choices, extra_frac, skew = PARAMS[tid - 1]

    seed = 900000 + tid * 97
    rng = random.Random(seed)

    # deterministic parameter nudges so the coupling graph is never fully
    # degenerate (every community reachable by some nonzero link, so the
    # instance genuinely exercises multi-round propagation).
    attempt = 0
    while True:
        local_rng = random.Random(seed + attempt * 131071)
        sizes = block_sizes(N, K, local_rng, skew)
        W = build_block_graph(K, local_rng, ring_w_choices, extra_frac, self_choices)
        nz_any = any(W[a][b] != 0 and mult(a, b, sizes) != 0
                      for a in range(K) for b in range(K))
        if nz_any or attempt > 8:
            break
        attempt += 1

    # community assignment: contiguous by size, then a deterministic shuffle
    c = []
    for a, sz in enumerate(sizes):
        c += [a] * sz
    local_rng.shuffle(c)
    assert len(c) == N

    out = []
    out.append(f"{N} {K} {CAP}")
    for a in range(K):
        out.append(" ".join(str(x) for x in W[a]))
    out.append(" ".join(str(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
