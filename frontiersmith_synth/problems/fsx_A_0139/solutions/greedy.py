# TIER: greedy
# Hammersley completion: fill the m new cameras with a standard 2D Hammersley set
#   x = (i+0.5)/m,  y = van-der-Corput radical inverse of i in base 2.
# Ignores the fixed cameras entirely, but a proper low-discrepancy fill reliably
# beats the concentrated diagonal baseline.
import sys

def vdc(i, base=2):
    f = 1.0; r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def main():
    toks = sys.stdin.read().split()
    M = int(toks[0]); k = int(toks[1])
    m = M - k
    out = []
    for i in range(m):
        x = (i + 0.5) / m
        y = vdc(i, 2)
        out.append("%.10f %.10f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
