import sys

# Independent oracle: trial division. Obviously correct, slow.
# Used only on numbers small enough that trial division finishes quickly.

def factor(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2  # after 2, only test odd divisors
    if n > 1:
        factors.append((n, 1))
    return factors

def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        n = int(data[idx]); idx += 1
        if n == 1:
            out.append("1:")
            continue
        fs = factor(n)
        parts = [f"{n}:"]
        for p, e in fs:
            parts.append(f"{p}^{e}")
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")

main()
