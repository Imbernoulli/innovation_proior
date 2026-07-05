# TIER: strong
# Halton sequence (radical-inverse in distinct prime bases). In d=2 the first coord uses
# a van der Corput sequence; for d=2 we replace it with the exact Hammersley coordinate
# i/M, which is the standard low-discrepancy construction for a fixed point count.
import sys

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]

def vdc(n, base):
    f = 1.0
    r = 0.0
    while n > 0:
        f /= base
        r += f * (n % base)
        n //= base
    return r

def main():
    inp = sys.stdin.read().split()
    d = int(inp[0]); M = int(inp[1])
    out = []
    for i in range(M):
        row = []
        for k in range(d):
            if k == 0:
                # Hammersley first coordinate for a fixed N (very low discrepancy).
                v = (i + 0.5) / M
            else:
                v = vdc(i + 1, PRIMES[k - 1])
            row.append("%.10f" % v)
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
