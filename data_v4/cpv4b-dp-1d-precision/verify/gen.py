#!/usr/bin/env python3
# Random small-case generator for "Cheapest pace relay": python3 gen.py <seed>
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(2, 9)
    mode = seed % 4
    def val():
        if mode == 0:
            return random.randint(1, 9)
        elif mode == 1:
            return random.randint(1, 40)
        elif mode == 2:
            return random.randint(1, 10**9)         # large -> stress cross-product magnitude
        else:
            # near-equal big ratios to force exact comparison
            return random.randint(10**9 - 30, 10**9)
    out = [str(n)]
    for i in range(n-1):
        out.append("%d %d" % (val(), val()))   # +1 hop from i
    for i in range(n-2):
        out.append("%d %d" % (val(), val()))   # +2 hop from i
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
