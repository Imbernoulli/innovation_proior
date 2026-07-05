# TIER: strong
# 2-D Hammersley point set: x_i = (i+0.5)/M, y_i = phi_2(i) (van der Corput
# radical inverse in base 2).  This is a classic low-discrepancy construction
# with O(log M / M) star discrepancy -- far below the regular grid, but it is
# NOT the (unknown) minimum-discrepancy layout, so headroom remains.
import sys


def radical_inverse_base2(i):
    result = 0.0
    f = 0.5
    while i > 0:
        result += (i & 1) * f
        i >>= 1
        f *= 0.5
    return result


def main():
    t = sys.stdin.read().split()
    d = int(t[0]); m = int(t[1])
    out = []
    for i in range(m):
        x = (i + 0.5) / m
        y = radical_inverse_base2(i)
        # nudge the y=0 point (i=0) off the boundary a hair so it samples interior
        if y == 0.0:
            y = 0.5 / m
        out.append("%.10f %.10f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
