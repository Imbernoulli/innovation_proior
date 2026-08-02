# TIER: trivial
# Do-nothing baseline: ignore Ta and the physics hints entirely, predict a
# single constant temperature -- the median of the observed training T
# column, never switching branches.  This reproduces the checker's own
# constant baseline -> Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("THRESH 1e9")
        print("BELOW 0.0")
        print("ABOVE 0.0")
        return
    n = int(data[0])
    rest = data[2:]  # skip "b h Hmax Tfail" header row (4 tokens)
    hints = rest[:4]
    rows = rest[4:]
    Ts = sorted(float(rows[2 * i + 1]) for i in range(n))
    med = Ts[n // 2] if n % 2 else 0.5 * (Ts[n // 2 - 1] + Ts[n // 2])
    print("THRESH 1e9")
    print("BELOW %.10g" % med)
    print("ABOVE %.10g" % med)


if __name__ == "__main__":
    main()
