# TIER: trivial
# Do-nothing baseline: predict a flat continuation of the mean of the last
# training week. Ignores growth, the market rhythm, the scribe cap, and the
# decree entirely. This reproduces the checker's own internal baseline.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0.0"); return
    n = int(data[0])
    rows = data[3:]
    r = [float(rows[2 * i + 1]) for i in range(n)]
    last7 = r[max(0, n - 7):n]
    mean_last7 = sum(last7) / len(last7)
    print("%.10f" % mean_last7)


if __name__ == "__main__":
    main()
