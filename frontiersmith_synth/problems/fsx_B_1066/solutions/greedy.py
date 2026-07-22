# TIER: greedy
"""Textbook circle-method round robin: N-1 rounds, each a perfect matching of
N/2 games, using all C=N/2 courts per date. This is the standard round-robin
construction anyone reaches for. It IS aware that unused slack dates exist
(so it spreads the N-1 rounds evenly across the full D-date calendar instead
of crunching them into the first N-1 dates), but it does this UNIFORMLY --
it never looks at which specific dates the K weather scenarios threaten, so
it cannot concentrate its scarce slack near the scenario-correlated clusters
that actually determine the worst case."""
import sys


def circle_rounds(N):
    teams = list(range(1, N + 1))
    fixed = teams[0]
    rot = teams[1:]
    rounds = []
    for _ in range(N - 1):
        arr = [fixed] + rot
        pairs = []
        for k in range(N // 2):
            a, b = arr[k], arr[N - 1 - k]
            pairs.append((min(a, b), max(a, b)))
        rounds.append(pairs)
        rot = [rot[-1]] + rot[:-1]
    return rounds


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); C = int(next(it))
    K = int(next(it)); next(it); next(it)
    for _ in range(K):
        b = int(next(it))
        for _ in range(b):
            next(it)

    rounds = circle_rounds(N)
    extra = D - (N - 1)
    dates = [1 + r + (r * extra) // (N - 1) for r in range(N - 1)]

    slot = {}
    for r, pairs in enumerate(rounds):
        d = dates[r]
        for k, (a, b) in enumerate(pairs):
            slot[(a, b)] = (d, k + 1)

    out = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            d, c = slot[(i, j)]
            out.append(f"{d} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
