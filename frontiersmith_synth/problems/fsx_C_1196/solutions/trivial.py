# TIER: trivial
"""
Trivial baseline: freeze adoption at the last observed training value and
predict that flat constant for every held-out step. This reproduces the
checker's own internal baseline construction, so it should land at
Ratio ~= 0.1.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    tid = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    tB = float(data[idx]); idx += 1
    s_hint = float(data[idx]); idx += 1
    M_hint = float(data[idx]); idx += 1
    A_last = None
    for _ in range(n):
        ti = float(data[idx]); idx += 1
        A = float(data[idx]); idx += 1
        A_last = A
    print("%.10g" % A_last)


if __name__ == "__main__":
    main()
