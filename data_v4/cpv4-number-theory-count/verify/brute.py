import sys
from math import gcd
from fractions import Fraction

def solve(n):
    seen = set()
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            # distinct rational value a/b, reduced canonical form
            g = gcd(a, b)
            seen.add((a // g, b // g))
    return len(seen)

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        out.append(str(solve(n)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
