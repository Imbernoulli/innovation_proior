# TIER: trivial
"""Naive constant-angular-step 'spokes': theta_k = k * (360/N). Ignores the
growth law entirely and reproduces the checker's own internal baseline."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    step = 360.0 / N
    out = []
    for k in range(1, N + 1):
        out.append("%.6f" % ((k * step) % 360.0))
    print("\n".join(out))


if __name__ == "__main__":
    main()
