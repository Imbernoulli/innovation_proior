# TIER: trivial
# Do-nothing baseline: predict a CONSTANT velocity equal to the mean of the
# training measurements. Ignores frequency entirely -> reproduces the
# checker's own internal baseline construction almost exactly (~0.1).
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    n = int(data[0])
    # data[0]=n, data[1]=t, data[2..5]=CA CB CD CE, then n (f v) pairs
    rows = data[6:]
    vals = [float(rows[2 * i + 1]) for i in range(n)]
    mean_v = sum(vals) / len(vals)
    print("%.6f" % mean_v)


if __name__ == "__main__":
    main()
