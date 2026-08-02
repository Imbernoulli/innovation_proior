# TIER: trivial
# Do-nothing baseline: predict the constant mean of the observed training
# errors, ignoring the decaying trend entirely. This reproduces the
# checker's own internal baseline construction exactly -> Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.5")
        return
    m = int(data[0])
    vals = data[2:]
    ys = [float(vals[2 * i + 1]) for i in range(m)]
    mean_y = sum(ys) / len(ys)
    print("%.10f" % mean_y)


if __name__ == "__main__":
    main()
