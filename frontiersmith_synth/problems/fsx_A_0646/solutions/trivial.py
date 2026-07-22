# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    agents = []
    for _ in range(n):
        p = float(data[idx]); a = float(data[idx + 1]); L = float(data[idx + 2]); u = float(data[idx + 3])
        idx += 4
        agents.append((p, a, L, u))

    if n == 0:
        print(0)
        return

    mean_pl = sum(p * L for (p, a, L, u) in agents) / n
    # naive: one full-coverage contract at a tiny 5% markup over the raw
    # population-mean expected loss -- ignores outside options entirely.
    P = 1.05 * mean_pl
    print(1)
    print("%.6f %.6f" % (P, 1.0))


if __name__ == "__main__":
    main()
