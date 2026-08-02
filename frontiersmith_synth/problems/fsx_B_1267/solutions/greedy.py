# TIER: greedy
"""The obvious first idea: treat plausibility as a per-claim fraud-risk
score and audit the LEAST plausible claims first, greedily filling the
budget. This is exactly a per-claim anomaly-ranking approach -- it will
catch lone, clumsy fraud (genuinely low plausibility) but has no way to
tell ring fraud apart from ordinary business, since ring claims were
engineered to sit in the same plausibility band as everything else."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it))
    next(it); next(it); next(it); next(it)  # NC NP NA testId
    plaus = []
    costs = []
    for _ in range(N):
        next(it); next(it); next(it); next(it)  # claimant provider adjuster amount
        plaus.append(float(next(it)))
        costs.append(int(next(it)))

    order = sorted(range(N), key=lambda i: (plaus[i], i))  # most suspicious first

    chosen = []
    used = 0
    for i in order:
        c = costs[i]
        if used + c <= M:
            chosen.append(i)
            used += c

    out = [str(len(chosen))] + [str(i) for i in chosen]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
