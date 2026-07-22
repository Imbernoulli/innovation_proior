# TIER: trivial
import sys


def main():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    n = int(data[0].split()[0])
    total = 0.0
    count = 0
    for line in data[1:1 + n]:
        parts = line.split()
        if len(parts) >= 5:
            total += float(parts[4])
            count += 1
    mean = total / max(1, count)
    sys.stdout.write("%r\n" % mean)


if __name__ == "__main__":
    main()
