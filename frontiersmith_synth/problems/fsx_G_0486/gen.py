#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Ternary-tagging cap-set instance. The instance is simply the tag length n
(the ambient dimension of F_3^n). Difficulty ladder: n grows with testId, so
the space (and the search) grows while the achievable cap density stays open.

The score depends ONLY on n (deterministic); no hidden data is emitted.
"""
import sys


def main():
    try:
        t = int(sys.argv[1])
    except Exception:
        t = 1
    if t < 1:
        t = 1
    # testId 1..4 -> n = 9,10,11,12  (all genuinely open: max cap unknown)
    n = t + 8
    sys.stdout.write("# ternary-tagging cap set instance\n")
    sys.stdout.write("%d\n" % n)


if __name__ == "__main__":
    main()
