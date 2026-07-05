# TIER: greedy
"""Greedy improvement: replace ONE {0,1}^4 block (16 tags) by the optimal
20-cap of F_3^4, keeping {0,1} on the remaining coordinates. A product of caps
is a cap, so this is valid. Size 20 * 2^(n-4) = 1.25 * 2^n -> Ratio ~ 0.125."""
import sys, itertools

CAP4 = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
        '1000', '1001', '1012', '1022', '1102', '1202', '2012', '2102',
        '2110', '2111', '2122', '2212']


def main():
    n = None
    for tok in sys.stdin.read().split():
        try:
            n = int(tok); break
        except ValueError:
            continue
    if n is None:
        n = 1
    if n < 4:
        out = ["".join(t) for t in itertools.product("01", repeat=n)]
    else:
        tails = ["".join(t) for t in itertools.product("01", repeat=n - 4)]
        out = [c + t for c in CAP4 for t in tails]
    buf = [str(len(out))]
    buf.extend(out)
    sys.stdout.write("\n".join(buf) + "\n")


if __name__ == "__main__":
    main()
