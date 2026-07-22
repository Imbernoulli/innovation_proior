# TIER: trivial
import sys
from fractions import Fraction as Fr


SCALE = 10 ** 9


def rnd(fr):
    # snap UP to the printable 1e-9 grid; used for every time value we ever emit or
    # build further arithmetic on, so our own bookkeeping matches EXACTLY what the
    # checker re-derives from the (grid-snapped) apply times we output.
    scaled = -(-(fr.numerator * SCALE) // fr.denominator)
    return Fr(scaled, SCALE)


def fmt(fr):
    # fr is assumed already grid-snapped (a multiple of 1/SCALE), so fr*SCALE is
    # exactly integral.
    iv = int(fr * SCALE)
    ip, fp = divmod(iv, SCALE)
    return "%d.%09d" % (ip, fp)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    cn = int(next(it))
    cd = int(next(it))
    c = Fr(cn, cd)
    bases = []
    for _p in range(P):
        k = int(next(it))
        for _j in range(k):
            bases.append(int(next(it)))

    # fully sequential: one coat wet at a time, everywhere, always.
    mult = (1 + c) * (1 + c)
    cur = Fr(0)
    out = []
    for b in bases:
        cur = rnd(cur)
        out.append(fmt(cur))
        cur = cur + Fr(b) * mult
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
