#!/usr/bin/env python3
# Random small-case generator.  Usage: gen.py <seed>
# Emits one token: a string over {a..z, '?'}.  Small alphabets and
# moderate '?'-rates are weighted in to stress the transitivity trap.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # length: mostly small to keep the O(n^2) brute fast
    mode = rng.randint(0, 5)
    if mode == 0:
        n = rng.randint(1, 3)
    elif mode == 1:
        n = rng.randint(1, 8)
    elif mode == 2:
        n = rng.randint(1, 20)
    elif mode == 3:
        n = rng.randint(1, 40)
    else:
        n = rng.randint(1, 80)

    # alphabet size: tiny alphabets create lots of near-periods + traps
    sigma = rng.choice([1, 2, 2, 3, 3, 4, 6])
    letters = "abcdefghijklmnopqrstuvwxyz"[:sigma]

    # wildcard probability: vary to make the transitive closure bite
    qrate = rng.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])

    chars = []
    for _ in range(n):
        if rng.random() < qrate:
            chars.append('?')
        else:
            chars.append(rng.choice(letters))

    # occasionally plant an exact period then corrupt one position
    if rng.random() < 0.35 and n >= 4:
        p = rng.randint(1, n)
        base = [rng.choice(letters) for _ in range(p)]
        chars = [base[i % p] for i in range(n)]
        # sprinkle some '?'
        for i in range(n):
            if rng.random() < qrate:
                chars[i] = '?'
        # maybe corrupt one concrete cell to break the period
        if rng.random() < 0.6:
            i = rng.randrange(n)
            chars[i] = rng.choice(letters)

    print(''.join(chars))

if __name__ == "__main__":
    main()
