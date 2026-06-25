import sys

# Independent, obviously-correct reference.
# D(m) = sum over t=1..m of (number of divisors of t),
# computed WITHOUT the hyperbola identity: directly add floor(m/i)
# for i = 1..m  (each i contributes the count of its multiples <= m,
# i.e. the number of t <= m divisible by i; summing over all divisors
# i gives the total divisor count). This is the plain definition.
def D(m):
    if m <= 0:
        return 0
    total = 0
    for i in range(1, m + 1):
        total += m // i
    return total

def smallest_m(K):
    # linear scan from m = 1 upward until D(m) >= K (small K only).
    m = 0
    val = 0
    # incremental: D(m) - D(m-1) = number of divisors of m
    while val < K:
        m += 1
        # number of divisors of m
        d = 0
        j = 1
        while j * j <= m:
            if m % j == 0:
                d += 1 if j * j == m else 2
            j += 1
        val += d
    return m

def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        K = int(data[idx]); idx += 1
        out.append(str(smallest_m(K)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
