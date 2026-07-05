# TIER: greedy
# 3D Halton set: coordinate j = radical inverse of the index in prime base b_j,
# with bases (2, 3, 5). A standard fixed low-discrepancy construction that spreads
# the quadrats through the whole environmental cube and reliably beats the diagonal.
import sys

def radical_inverse(i, base):
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
    bases = (2, 3, 5)
    out = []
    for i in range(n):
        # start at index 1 so no point sits exactly at the origin
        idx = i + 1
        x = radical_inverse(idx, bases[0])
        y = radical_inverse(idx, bases[1])
        z = radical_inverse(idx, bases[2])
        out.append("%.10f %.10f %.10f" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
