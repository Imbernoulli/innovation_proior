# TIER: trivial
# Do-nothing-clever baseline: fit a PLAIN PROPORTIONAL throttle law
# u = Kp0 * e by ordinary least squares on the calm log, ignoring the
# accumulated-error register entirely (no memory, no saturation).  This is
# exactly the construction the checker uses as its own internal baseline, so
# it reproduces Ratio ~= 0.1 by design.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.0 * e")
        return
    n = int(data[0])
    vals = data[3:]
    num = 0.0
    den = 0.0
    for i in range(n):
        sp = float(vals[3 * i])
        y = float(vals[3 * i + 1])
        u = float(vals[3 * i + 2])
        e = sp - y
        num += e * u
        den += e * e
    Kp0 = num / den if den > 1e-9 else 0.0
    print("OUT %.6f * e" % Kp0)


if __name__ == "__main__":
    main()
