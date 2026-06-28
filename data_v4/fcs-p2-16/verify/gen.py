#!/usr/bin/env python3
"""Random + edge-case test generator for the huge-capacity 0/1 knapsack problem.

Usage:
    gen.py SEED [MODE]

Emits one test case to stdout in the format:
    n C
    w_i v_i   (n lines)

MODE controls the flavor of the instance. Because the brute oracle is 2^n,
generated n is kept small (<= 18). The cases are crafted to stress the places
where a greedy-by-ratio or sloppy branch-and-bound would go wrong:
  - tight capacities where ratio greedy mispacks,
  - capacities far above the total weight (answer = sum of all values),
  - capacity 0, single items, all-too-heavy items,
  - large weight/value magnitudes to exercise 64-bit arithmetic and
    the split/binary-search boundary.
"""
import random
import sys


def emit(n, C, items):
    out = [f"{n} {C}"]
    for (w, v) in items:
        out.append(f"{w} {v}")
    sys.stdout.write("\n".join(out) + "\n")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "edge":
        # A rotating set of deliberate corner cases keyed by seed.
        cases = []

        # n = 1, capacity too small (cannot take the item).
        cases.append((1, 5, [(10, 1000000000)]))
        # n = 1, capacity exactly the weight.
        cases.append((1, 10, [(10, 7)]))
        # capacity 0: nothing fits, answer 0.
        cases.append((3, 0, [(1, 5), (2, 9), (3, 1)]))
        # capacity huge (>= total weight): take everything.
        cases.append((4, 10**18, [(10**9, 10**9), (10**9, 10**9),
                                  (10**9, 10**9), (10**9, 10**9)]))
        # classic ratio-greedy trap: one high-ratio light item blocks a
        # better pair (3,7) vs (4,5)+(4,5)? craft directly:
        # capacity 8; items (5,11) ratio 2.2, (4,7) ratio 1.75, (4,7).
        # greedy by ratio takes (5,11) then nothing else fits in 3 -> 11,
        # but (4,7)+(4,7)=14 fits in 8.
        cases.append((3, 8, [(5, 11), (4, 7), (4, 7)]))
        # all items individually too heavy.
        cases.append((3, 4, [(5, 100), (6, 200), (7, 300)]))
        # mix near the 64-bit boundary, capacity between two thresholds.
        cases.append((5, 3 * 10**9, [(10**9, 1), (10**9, 1), (10**9, 1),
                                     (10**9, 5), (10**9, 5)]))
        # n at the split boundary (split is n/2): n = 2,3 small smoke tests.
        cases.append((2, 1, [(1, 1), (1, 1)]))
        cases.append((3, 2, [(1, 1), (1, 1), (1, 1)]))

        c = cases[seed % len(cases)]
        emit(c[0], c[1], c[2])
        return

    # Random mode.
    n = rng.randint(0, 18)
    items = []
    # Pick a magnitude scale so we sometimes get tiny weights, sometimes huge.
    scale = rng.choice([10, 100, 10**6, 10**9])
    for _ in range(n):
        w = rng.randint(1, scale)
        # Values sometimes correlated with weight, sometimes anti-correlated,
        # to break ratio heuristics.
        vscale = rng.choice([10, 100, 10**6, 10**9])
        v = rng.randint(1, vscale)
        items.append((w, v))

    total_w = sum(w for (w, _) in items)
    # Choose capacity in a regime that actually constrains the answer.
    ctype = rng.choice(["tiny", "tight", "mid", "huge", "zero"])
    if n == 0 or total_w == 0:
        C = rng.randint(0, 10**18)
    elif ctype == "zero":
        C = 0
    elif ctype == "tiny":
        C = rng.randint(0, max(1, total_w // (2 * n) if n else 1))
    elif ctype == "tight":
        C = rng.randint(total_w // 4, max(total_w // 4, total_w // 2))
    elif ctype == "mid":
        C = rng.randint(total_w // 2, total_w)
    else:  # huge
        C = rng.randint(total_w, 10**18)

    emit(n, C, items)


if __name__ == "__main__":
    main()
