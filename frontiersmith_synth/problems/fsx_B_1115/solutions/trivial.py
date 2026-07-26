# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    times = [int(next(it)) for _ in range(T)]
    demands = [int(next(it)) for _ in range(T)]
    S1 = int(next(it)); S2 = int(next(it)); S3 = int(next(it))
    d1 = float(next(it)); d2 = float(next(it)); d3 = float(next(it))

    # one-lot-per-pulse: every stage produces & consumes in the same hour as the
    # pulse it serves, so zero decay is ever incurred (but max setup scrap).
    lines = []
    N = 3 * T
    lines.append(str(N))
    for j in range(T):
        t = times[j]; D = demands[j]
        r3 = D + S3
        r2 = D + S3 + S2
        r1 = D + S3 + S2 + S1
        lines.append("1 %d %.6f" % (t, r1))
        lines.append("2 %d %.6f" % (t, r2))
        lines.append("3 %d %.6f" % (t, r3))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
