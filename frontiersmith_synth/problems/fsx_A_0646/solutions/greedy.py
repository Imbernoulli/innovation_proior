# TIER: greedy
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

    # textbook actuarial recipe: one full-coverage contract priced at a flat
    # markup over the population-mean expected loss. Never looks at u_i, never
    # asks who would actually walk away -- vulnerable to adverse selection
    # whenever risk correlates with participation across a hidden mixture.
    mean_pl = sum(p * L for (p, a, L, u) in agents) / n
    P = 1.20 * mean_pl
    print(1)
    print("%.6f %.6f" % (P, 1.0))


if __name__ == "__main__":
    main()
