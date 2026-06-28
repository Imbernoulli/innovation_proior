#!/usr/bin/env python3
"""Instance generator for "Simulated Epidemic Containment" (ALE-Bench heuristic
optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format (all integers / fixed-precision reals,
whitespace separated):

    n m T b
    beta gamma kappa            (three reals: SIR rates / coupling, 6 decimals)
    u_0 v_0 w_0                 \
    ...                          |  m undirected weighted edges (region graph)
    u_{m-1} v_{m-1} w_{m-1}     /   0 <= u,v < n, u != v, w in (0,1] (6 decimals)
    I0_0 I0_1 ... I0_{n-1}      (n reals in [0,1]: initial infected fraction / region)

Meaning. There are `n` regions. Each region r holds three compartment fractions
S_r + I_r + R_r = 1 (susceptible / infected / recovered), evolving over `T`
discrete days by a deterministic SIR-on-a-graph model. Each day, BEFORE the spread
step, the controller may LOCK DOWN up to `b` regions; a locked region's
transmission (internal and across its incident edges) is multiplied by `kappa`
(a residual-transmission factor in [0,1)) for that day. The objective is to choose
the per-day lockdown sets to MINIMIZE the total number of NEW infections summed
over all regions and all T days (see score.py / context.md for the exact dynamics,
the scoring rule, and the feasibility -> 0 floor).

Instance regime (deterministic from the seed):
  * n regions in [30, 60]; the contact graph is connected (a random spanning tree
    plus extra random edges), average degree ~3-5; edge weights w in (0,1].
  * T days in [16, 24]; daily lockdown budget b in [4, 6] (so b << n: you can only
    cover a small fraction of regions per day, the budget is BINDING).
  * beta (transmission) in [0.26, 0.40], gamma (recovery) in [0.12, 0.18] -> the
    basic reproduction number sits around 1.5-2.5: an uncontrolled epidemic still
    grows, but the budget is large enough that smart containment can genuinely
    SUPPRESS it (not merely delay it). This is the regime where the per-day
    lockdown CHOICE actually changes the cumulative attack rate -- if R0 were far
    above 1 with a long horizon the epidemic would saturate no matter what, and
    the schedule would not matter.
  * kappa (residual transmission of a locked region) in [0.08, 0.22] -> a lockdown
    is strong but not perfect.
  * A few seed regions start with a small infected fraction; the rest start almost
    fully susceptible. The seeds are placed so the epidemic must travel across the
    graph, which is what makes a LOOK-AHEAD (where will infection be in k days)
    beat a myopic "lock the most-infected-today" rule.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x40E9_1D00 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(30, 60)
    T = rng.randint(16, 24)
    b = rng.randint(4, 6)

    beta = rng.uniform(0.26, 0.40)
    gamma = rng.uniform(0.12, 0.18)
    kappa = rng.uniform(0.08, 0.22)

    # ---- contact graph: random spanning tree (connected) + extra edges ----
    edges = {}

    def add_edge(u, v, w):
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        if key in edges:
            return
        edges[key] = w

    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(1, n):
        u = perm[i]
        v = perm[rng.randrange(i)]  # connect to an earlier node -> tree
        add_edge(u, v, round(rng.uniform(0.4, 1.0), 6))
    # extra edges to raise average degree
    extra = rng.randint(n // 2, n)
    for _ in range(extra):
        u = rng.randrange(n)
        v = rng.randrange(n)
        add_edge(u, v, round(rng.uniform(0.2, 1.0), 6))

    edge_list = [(u, v, w) for (u, v), w in edges.items()]
    rng.shuffle(edge_list)
    m = len(edge_list)

    # ---- initial infections: a few seed regions ----
    I0 = [0.0] * n
    nseed = rng.randint(1, 3)
    seeds = rng.sample(range(n), nseed)
    for s in seeds:
        I0[s] = round(rng.uniform(0.02, 0.08), 6)

    out = []
    out.append(f"{n} {m} {T} {b}")
    out.append(f"{beta:.6f} {gamma:.6f} {kappa:.6f}")
    for (u, v, w) in edge_list:
        out.append(f"{u} {v} {w:.6f}")
    out.append(" ".join(f"{x:.6f}" for x in I0))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
