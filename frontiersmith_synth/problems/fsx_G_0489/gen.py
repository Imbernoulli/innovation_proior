#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE instance (window width W) to stdout.
# testId 1..10 is a small->large difficulty ladder; deterministic, no randomness.
import sys

WS = [120, 250, 450, 750, 1100, 1600, 2300, 3200, 4300, 5600]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(WS):
        # extend deterministically beyond the ladder if ever requested
        W = WS[-1] + (t - len(WS)) * 900
    else:
        W = WS[t - 1]
    print(W)


if __name__ == "__main__":
    main()
