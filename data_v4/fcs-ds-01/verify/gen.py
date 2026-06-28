#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
#
# Emits a valid instance of the offline range-distinct problem:
#   n q
#   a[0..n-1]
#   q lines of "l r"  (0 <= l <= r < n)
#
# Value ranges are biased small so ranges share many duplicates (this exercises
# the count-array add/remove transitions where a value's count hits or leaves
# zero -- the part of Mo's algorithm most likely to be wrong). Degenerate cases
# (n == 1, single-element ranges, full-array ranges) appear with nonzero
# probability.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 30)
    q = rng.randint(1, 30)

    # Bias the value alphabet small to force duplicates and heavy overlap.
    vmax = rng.choice([1, 1, 2, 3, 5, 10, 50, 1000000000])
    # Occasionally allow negative values to test that the solution does not
    # assume non-negativity (compression must still work).
    if rng.random() < 0.3:
        a = [rng.randint(-vmax, vmax) for _ in range(n)]
    else:
        a = [rng.randint(1, vmax) for _ in range(n)]

    lines = ["%d %d" % (n, q), " ".join(map(str, a))]
    for _ in range(q):
        l = rng.randint(0, n - 1)
        r = rng.randint(0, n - 1)
        if l > r:
            l, r = r, l
        # ~20% of the time force a single-element range (l == r).
        if rng.random() < 0.2:
            r = l
        lines.append("%d %d" % (l, r))

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
