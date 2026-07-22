# TIER: trivial
# Do-nothing baseline: a single geometric term extrapolated from the last two
# visible points (exactly the checker's own internal baseline construction).
# Ignores every other visible point and any higher-order structure -> ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    T, t = int(data[0]), int(data[1])
    rows = data[2:]
    a = {}
    for i in range(T):
        n = int(rows[2 * i])
        v = int(rows[2 * i + 1])
        a[n] = v
    last1 = a[T - 1]
    last2 = a[T - 2]
    ratio = last1 / last2
    A = last1 / (ratio ** (T - 1))
    print("( %r ) * ( %r ) ** n" % (A, ratio))


if __name__ == "__main__":
    main()
