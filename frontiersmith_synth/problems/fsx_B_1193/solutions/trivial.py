# TIER: trivial
# Do-nothing baseline: predict the constant mean training force, ignoring x
# and b entirely -> reproduces the checker's own constant baseline (~0.1).
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    vals = data[2:]
    ys = [float(vals[2 * i + 1]) for i in range(n)]
    mean_y = sum(ys) / len(ys) if ys else 0.0
    print("%.6f" % mean_y)


if __name__ == "__main__":
    main()
