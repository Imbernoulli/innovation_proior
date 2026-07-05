#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance (number of cave segments n) to stdout.
# testId 1..10 = difficulty ladder (small -> large n).
import sys

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    # large-scale ladder: n = 120, 160, ..., 480 (all even -> clean baseline)
    n = 80 + 40 * t
    print(n)

if __name__ == "__main__":
    main()
