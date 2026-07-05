# TIER: trivial
# Naive "ramp up to a midday peak, then ramp down" schedule (triangular).
# This is exactly the checker's internal baseline -> scores ~0.1.
import sys

def main():
    n, V = map(int, sys.stdin.read().split())
    a = []
    for i in range(n):
        frac = 1.0 - abs(2.0 * i / (n - 1) - 1.0)
        a.append(int(round(V * frac)))
    if sum(a) == 0:
        a[0] = V
    sys.stdout.write(" ".join(map(str, a)) + "\n")

if __name__ == "__main__":
    main()
