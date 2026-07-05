# TIER: trivial
"""Predict the constant mean training calibration error -> reproduces baseline."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    hdr = data[0].split()
    n = int(hdr[0])
    ys = []
    for i in range(1, n + 1):
        parts = data[i].split()
        ys.append(float(parts[2]))
    m = sum(ys) / len(ys)
    sys.stdout.write("%r\n" % m)


if __name__ == "__main__":
    main()
