# TIER: strong
"""Strong construction: tile the coordinates into as many disjoint {0,1}^4 -> 20-cap
blocks as fit (floor(n/4) of them), with {0,1} on any leftover coordinates. The
product of caps is a cap, so it is valid. Size 20^(n//4) * 2^(n mod 4), giving
Ratio 0.1 * 1.25^(n//4) -- strictly denser than the single-block greedy, and it
improves further whenever another full block of 4 coordinates becomes available.
Still far below the (unknown) true maximum cap: genuine headroom remains."""
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
    b = n // 4
    r = n - 4 * b
    if b == 0:
        out = ["".join(t) for t in itertools.product("01", repeat=n)]
    else:
        blocks = [CAP4] * b + [["0", "1"]] * r
        out = ["".join(combo) for combo in itertools.product(*blocks)]
    buf = [str(len(out))]
    buf.extend(out)
    sys.stdout.write("\n".join(buf) + "\n")


if __name__ == "__main__":
    main()
