#!/usr/bin/env python3
# Random small-case generator for fcs-ds-03.  Usage: python3 gen.py <seed>
# Emits:  n ; a[1..n] ; b[0..n-1] ; c[1..n]
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes selected by the seed to stress different envelope shapes.
    mode = seed % 5
    if mode == 0:                       # tiny, full range incl. negatives
        n = rng.randint(0, 6)
        V = 50
    elif mode == 1:                     # small, only positive (monotone-friendly)
        n = rng.randint(1, 12)
        V = 30
    elif mode == 2:                     # medium with negatives
        n = rng.randint(1, 40)
        V = 100
    elif mode == 3:                     # many ties in a[] (compression stress)
        n = rng.randint(1, 40)
        V = 4
    else:                               # larger small case
        n = rng.randint(1, 80)
        V = 1000

    out = [str(n)]
    if n > 0:
        a = [rng.randint(-V, V) for _ in range(n)]      # a[1..n]
        b = [rng.randint(-V, V) for _ in range(n)]      # b[0..n-1]
        c = [rng.randint(-V, V) for _ in range(n)]      # c[1..n]
        out.append(' '.join(map(str, a)))
        out.append(' '.join(map(str, b)))
        out.append(' '.join(map(str, c)))
    sys.stdout.write('\n'.join(out) + '\n')

main()
