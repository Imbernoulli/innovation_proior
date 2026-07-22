#!/usr/bin/env python3
# Generator for "Long Square-Free Word Hitting a Target Letter Mix" (format C).
# python3 gen.py <testId> -> prints ONE instance to stdout. Deterministic in testId only.
import sys


def main():
    tid = int(sys.argv[1])

    # difficulty ladder: length grows with testId; target letter-frequency mix
    # (w0,w1,w2,w3, integers /10000) gets progressively more skewed toward letter 0.
    Ls = [100, 120, 150, 180, 200, 240, 280, 320, 380, 450]
    Ws = [
        (2700, 2500, 2450, 2350),
        (3200, 2600, 2200, 2000),
        (4000, 2200, 2000, 1800),
        (2800, 2500, 2400, 2300),
        (4500, 2000, 1800, 1700),
        (3000, 2500, 2300, 2200),
        (3500, 2500, 2200, 1800),
        (5000, 1800, 1700, 1500),
        (2900, 2500, 2350, 2250),
        (4200, 2100, 2000, 1700),
    ]
    # target transition preference: succ[i] = the letter that should ideally follow
    # letter i.  Fixed derangement (a cyclic "shift by -1": 0<-1<-2<-3<-0) for every
    # test -- it is handed to the solver explicitly via the input, never hidden.
    Succ = (3, 0, 1, 2)

    idx = tid - 1
    if idx < 0 or idx >= len(Ls):
        # extend deterministically beyond the shipped ladder if ever asked for more
        idx = idx % len(Ls)
    L = Ls[idx]
    w = Ws[idx]

    print(L)
    print(w[0], w[1], w[2], w[3])
    print(Succ[0], Succ[1], Succ[2], Succ[3])


if __name__ == "__main__":
    main()
