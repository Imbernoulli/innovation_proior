# TIER: greedy
# 3D Hammersley set: coord0 = (i+0.5)/n, coord1 = radical inverse base 2,
# coord2 = radical inverse base 3. A standard low-discrepancy construction that
# reliably beats the diagonal baseline.
import sys


def vdc(i, base):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        x = (i + 0.5) / n
        y = vdc(i, 2)
        z = vdc(i, 3)
        out.append("%.10f %.10f %.10f" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
