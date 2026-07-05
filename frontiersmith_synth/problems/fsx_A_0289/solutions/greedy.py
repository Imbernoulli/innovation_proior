# TIER: greedy
# Kronecker / additive-recurrence lattice: x_i^k = frac((i+1) * sqrt(prime_k)).
# A decent low-discrepancy sequence, but generally weaker than Halton.
import sys, math

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]

def main():
    inp = sys.stdin.read().split()
    d = int(inp[0]); M = int(inp[1])
    alpha = [math.sqrt(PRIMES[k]) % 1.0 for k in range(d)]
    out = []
    for i in range(M):
        row = []
        for k in range(d):
            v = ((i + 1) * alpha[k]) % 1.0
            row.append("%.10f" % v)
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
