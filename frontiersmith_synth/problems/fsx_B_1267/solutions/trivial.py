# TIER: trivial
"""Ignore every feature. Walk claims in the order they were given and audit
whichever ones still fit the remaining budget. No use of amount,
plausibility, or party identity at all."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it))
    next(it); next(it); next(it); next(it)  # NC NP NA testId
    costs = []
    for _ in range(N):
        next(it); next(it); next(it); next(it); next(it)  # claimant provider adjuster amount plausibility
        costs.append(int(next(it)))

    chosen = []
    used = 0
    for i in range(N):
        c = costs[i]
        if used + c <= M:
            chosen.append(i)
            used += c

    out = [str(len(chosen))] + [str(i) for i in chosen]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
