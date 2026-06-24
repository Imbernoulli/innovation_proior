#!/usr/bin/env python3
# Random SMALL-case generator parameterized by an integer seed:  python3 gen.py <seed>
# Prints two lines: s then t. Small alphabets and short lengths so the brute force is fast and
# so feasible tilings (and greedy traps) actually occur frequently.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Bias toward a small alphabet so prefixes of s actually re-match inside t.
    alpha_size = rng.choice([2, 2, 2, 3])
    alphabet = "abc"[:alpha_size]

    m = rng.randint(1, 6)
    s = "".join(rng.choice(alphabet) for _ in range(m))

    # Two regimes for t:
    #  (A) totally random t (often infeasible -> exercises the -1 path)
    #  (B) t built by concatenating random prefixes of s (always feasible -> exercises counting)
    regime = rng.randint(0, 2)
    if regime == 0:
        n = rng.randint(1, 12)
        t = "".join(rng.choice(alphabet) for _ in range(n))
    else:
        blocks = rng.randint(1, 5)
        parts = []
        for _ in range(blocks):
            L = rng.randint(1, m)
            parts.append(s[:L])
        t = "".join(parts)
        # occasionally perturb one character so feasibility is not guaranteed
        if regime == 1 and t and rng.random() < 0.35:
            pos = rng.randrange(len(t))
            t = t[:pos] + rng.choice(alphabet) + t[pos+1:]

    # The contract requires a non-empty t is allowed to be empty too; brute handles empty.
    print(s)
    print(t)

if __name__ == "__main__":
    main()
