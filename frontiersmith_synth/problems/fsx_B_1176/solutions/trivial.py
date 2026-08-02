# TIER: trivial
"""Reproduces the checker's own internal baseline exactly: every appliance
cycles at its own MINIMUM legal dwell (min-off, then min-on, repeating),
completely ignoring the observed aggregate trace. Always legal (every
interior run sits exactly at the min bound, which is within [min,max])."""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr]); A = int(data[ptr + 1]); ptr += 2
    ptr += 2  # wT wA unused
    archs = []
    for _ in range(A):
        P, mon, mxon, moff, mxoff = (int(x) for x in data[ptr:ptr + 5]); ptr += 5
        archs.append((P, mon, mxon, moff, mxoff))
    # aggregate trace intentionally unused

    out = [str(A)]
    for (P, mon, mxon, moff, mxoff) in archs:
        seq = []
        state = 0
        while len(seq) < T:
            d = moff if state == 0 else mon
            d = min(d, T - len(seq))
            seq.extend([state] * d)
            state = 1 - state
        out.append(" ".join(str(v) for v in seq[:T]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
