# TIER: greedy
# Hammersley set: x = (i+0.5)/n, y = radical inverse of i in base 2. A standard
# low-discrepancy construction; reliably beats the diagonal baseline.
import sys

def vdc(i, base=2):
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
        out.append("%.10f %.10f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
