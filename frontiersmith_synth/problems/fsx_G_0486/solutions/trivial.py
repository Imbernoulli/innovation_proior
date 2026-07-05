# TIER: trivial
"""Trivial cap: the {0,1}^n grid. It is a valid cap (any collinear triple would
force a+b+c==0 in every coordinate, which for entries in {0,1} forces a=b=c).
Size 2^n = exactly the checker baseline -> Ratio ~ 0.1."""
import sys, itertools


def main():
    n = None
    for tok in sys.stdin.read().split():
        try:
            n = int(tok); break
        except ValueError:
            continue
    if n is None:
        n = 1
    out = ["".join(t) for t in itertools.product("01", repeat=n)]
    buf = [str(len(out))]
    buf.extend(out)
    sys.stdout.write("\n".join(buf) + "\n")


if __name__ == "__main__":
    main()
