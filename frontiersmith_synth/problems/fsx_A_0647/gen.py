#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Instance = a single non-negative integer T. The target shape is the solid
rectangle of width k = max(1, bit_length(T)) and height T+1 (rows 0..T),
which must be assembled EXACTLY (no more, no less) and then stop growing.

Deterministic: seeded purely by testId via a fixed difficulty/trap ladder.
Small T (testId 1-3) are cheap sanity/edge cases. Large T (testId 6-10,
T in [24,39]) are the planted trap cases: they are large enough that the
naive per-row "unary" recipe (T+2 tile types) is clearly beaten by a
cooperative binary-counter machine (~5*bit_length(T) tile types), while
staying just small enough that the counter construction does not saturate
the checker's score cap.
"""
import sys

T_LIST = [0, 3, 7, 15, 20, 24, 28, 30, 33, 34]


def main():
    tid = int(sys.argv[1])
    idx = (tid - 1) % len(T_LIST)
    T = T_LIST[idx]
    print(T)


if __name__ == "__main__":
    main()
