#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance (n q) to stdout.

Difficulty ladder (small=fast reward, large=eval), fully determined by testId:
    1,2   -> n=4     3,4,5 -> n=5     6,7,8 -> n=6     9,10  -> n=7
q is always 3 (three phase programs per intersection)."""
import sys

LADDER = {1: 4, 2: 4, 3: 5, 4: 5, 5: 5, 6: 6, 7: 6, 8: 6, 9: 7, 10: 7}


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid in LADDER:
        n = LADDER[tid]
    else:
        # deterministic fallback for any out-of-range id
        n = 4 + ((tid - 1) % 4)
    sys.stdout.write("%d %d\n" % (n, 3))


if __name__ == "__main__":
    main()
