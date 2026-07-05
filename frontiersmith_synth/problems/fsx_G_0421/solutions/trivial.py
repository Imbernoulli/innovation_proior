# TIER: trivial
# Constant predictor: emit the mean training loss.  Reproduces the checker's
# constant-baseline -> Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    tot = 0.0
    cnt = 0
    for line in data[1:1 + n]:
        parts = line.split()
        if len(parts) < 3:
            continue
        tot += float(parts[2])
        cnt += 1
    mean = tot / max(1, cnt)
    sys.stdout.write("%r\n" % mean)


if __name__ == "__main__":
    main()
