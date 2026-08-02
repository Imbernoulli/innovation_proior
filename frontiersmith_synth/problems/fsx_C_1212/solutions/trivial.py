# TIER: trivial
"""Do-nothing baseline: ignore density and perturbation entirely and predict
the CONSTANT mean flow observed in training. This is exactly the checker's
own internal baseline construction, so it reproduces Ratio ~= 0.10."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # data[1] is the test id, then n triples (rho, P, q)
    total = 0.0
    idx = 2
    for _ in range(n):
        q = float(data[idx + 2])
        idx += 3
        total += q

    m = total / n
    print("%.6f" % m)


if __name__ == "__main__":
    main()
