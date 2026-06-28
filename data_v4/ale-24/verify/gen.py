#!/usr/bin/env python3
"""Instance generator for "Machine Assignment with Sequence-Dependent Setups"
(ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format (see context.md "Input / output
contract"):

    n M T
    d_0 d_1 ... d_{n-1}          (n job durations, one line)
    c_0 c_1 ... c_{n-1}          (n job types,     one line, each in [0, T))
    init_0 init_1 ... init_{T-1} (T initial setup costs, one line)
    s[0][0] s[0][1] ... s[0][T-1]   (T lines, the T x T setup matrix)
    ...
    s[T-1][0] ... s[T-1][T-1]

Meaning:
  * We have M identical parallel machines and n jobs. Job j has processing
    duration d_j and a type c_j in [0, T).
  * Each machine processes the jobs assigned to it in some order. Before the
    FIRST job of a machine (type a) the machine pays an initial setup init[a].
    Between two consecutive jobs of types a then b on the same machine, a setup
    s[a][b] is paid (a sequence-dependent / type-dependent changeover).
  * The load of a machine is the sum of its job durations plus all its setups;
    the objective (see score.py) is the TOTAL over all machines of (durations +
    setups). Lower is better.

Instance regime (deterministic from the seed):
  * n jobs in [120, 260], M machines in [4, 10], T types in [4, 9].
  * Durations d_j in [10, 100].
  * The setup matrix is asymmetric and NON-metric in general: same-type
    changeovers are cheap (s[a][a] small), cross-type changeovers are moderate
    to large, so grouping same-type jobs together on a machine and ordering the
    type-runs well (a "TSP on types" per machine) is what saves cost. init[a]
    is a moderate one-time cost.
  * Types are drawn with a skewed multiplicity so some types are common and some
    rare -- this makes the assignment+ordering non-trivial (a plain round-robin
    that ignores types pays a setup on almost every adjacency).
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5E70_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(120, 260)
    M = rng.randint(4, 10)
    T = rng.randint(4, 9)

    # Durations.
    d = [rng.randint(10, 100) for _ in range(n)]

    # Skewed type weights: a few common types, some rare.
    weights = [rng.randint(1, 10) ** 2 for _ in range(T)]
    types = rng.choices(range(T), weights=weights, k=n)

    # Initial setups: moderate one-time cost per type.
    init = [rng.randint(15, 60) for _ in range(T)]

    # T x T setup matrix. Same-type cheap; cross-type moderate-to-large; the
    # matrix is asymmetric (s[a][b] != s[b][a] in general) and non-metric.
    s = [[0] * T for _ in range(T)]
    for a in range(T):
        for b in range(T):
            if a == b:
                s[a][b] = rng.randint(0, 8)        # cheap same-type changeover
            else:
                base = rng.randint(20, 90)         # moderate-to-large cross-type
                # occasional "expensive" pairs to make ordering matter
                if rng.random() < 0.18:
                    base += rng.randint(40, 120)
                s[a][b] = base

    out = []
    out.append(f"{n} {M} {T}")
    out.append(" ".join(map(str, d)))
    out.append(" ".join(map(str, types)))
    out.append(" ".join(map(str, init)))
    for a in range(T):
        out.append(" ".join(map(str, s[a])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
