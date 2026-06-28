#!/usr/bin/env python3
"""
Random small-case generator for the de Bruijn reconstruction problem.

  gen.py <seed>

Mixes several regimes so that both solvable and unsolvable instances appear:
  - "from-string": pick a random short string, emit ALL its k-mers (always
    solvable, possibly with repeats / branching de Bruijn structure).
  - "random-kmers": emit independently random k-mers (often unsolvable;
    stresses the existence test + IMPOSSIBLE path).
  - shuffled order, small alphabet to force overlaps and repeated edges.
"""
import sys
import random


def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)

    regime = rng.randrange(3)
    alpha_size = rng.choice([2, 2, 3, 4])
    alphabet = "abcdefghij"[:alpha_size]
    k = rng.randint(2, 4)

    kmers = []
    if regime == 0:
        # from a random string: all windows (guaranteed solvable)
        L = rng.randint(k, k + 6)
        s = "".join(rng.choice(alphabet) for _ in range(L))
        kmers = [s[i:i + k] for i in range(L - k + 1)]
        rng.shuffle(kmers)
    elif regime == 1:
        # purely random k-mers (often IMPOSSIBLE)
        m = rng.randint(1, 7)
        kmers = ["".join(rng.choice(alphabet) for _ in range(k)) for _ in range(m)]
    else:
        # from-string but then add/remove a few k-mers to perturb solvability
        L = rng.randint(k, k + 5)
        s = "".join(rng.choice(alphabet) for _ in range(L))
        kmers = [s[i:i + k] for i in range(L - k + 1)]
        # perturb
        ops = rng.randint(0, 3)
        for _ in range(ops):
            if kmers and rng.random() < 0.5:
                kmers.pop(rng.randrange(len(kmers)))
            else:
                kmers.append("".join(rng.choice(alphabet) for _ in range(k)))
        rng.shuffle(kmers)

    # occasionally emit an empty list
    if rng.random() < 0.03:
        kmers = []

    m = len(kmers)
    out = [f"{k} {m}"]
    out.extend(kmers)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
