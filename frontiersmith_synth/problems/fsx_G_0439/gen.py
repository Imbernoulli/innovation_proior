#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance (difficulty ladder, seeded by testId only)."""
import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(1000 + tid)
    # difficulty grows with testId: more targets, larger values
    k = 4 + 4 * tid                       # tid1 -> 8, tid10 -> 44
    hi = min(4000, 220 + 380 * tid)       # tid1 -> 600, tid10 -> 4000
    lo = max(3, hi // 10)
    targets = set()
    while len(targets) < k:
        targets.add(rng.randint(lo, hi))
    targets = sorted(targets)
    print(k)
    print(' '.join(map(str, targets)))

if __name__ == "__main__":
    main()
