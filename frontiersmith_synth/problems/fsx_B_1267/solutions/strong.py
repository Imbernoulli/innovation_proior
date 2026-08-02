# TIER: strong
"""The insight: score GROUPS, not claims. A collusion ring is invisible in
any single claim's plausibility -- it is only visible as a (claimant,
provider)/(claimant,adjuster)/(provider,adjuster) party PAIR that recurs
across many claims, because the ring keeps reusing the same tiny pool of
colluding parties. Build the bipartite claim<->party incidence, count how
often each unordered party pair co-occurs across claims, and give every
claim a "recurrence score" = the highest co-occurrence count among its
three party pairs (excluding itself). Claims from a genuine ring share a
pair that appears many times; ordinary and lone-sloppy claims draw their
parties from a wide pool, so their pairs are essentially unique.

Spend the budget on the highest-recurrence claims first (breaking ties by
value density), and only once that structural signal is exhausted fall
back to the plausibility outlier heuristic for whatever budget remains."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it))
    next(it); next(it); next(it); next(it)  # NC NP NA testId

    claimant = [0] * N
    provider = [0] * N
    adjuster = [0] * N
    amount = [0.0] * N
    plaus = [0.0] * N
    costs = [0] * N
    for i in range(N):
        claimant[i] = int(next(it))
        provider[i] = int(next(it))
        adjuster[i] = int(next(it))
        amount[i] = float(next(it))
        plaus[i] = float(next(it))
        costs[i] = int(next(it))

    def keys_of(i):
        return (("CP", claimant[i], provider[i]),
                ("CA", claimant[i], adjuster[i]),
                ("PA", provider[i], adjuster[i]))

    pair_count = {}
    for i in range(N):
        for k in keys_of(i):
            pair_count[k] = pair_count.get(k, 0) + 1

    ring_score = [0] * N
    for i in range(N):
        ring_score[i] = max(pair_count[k] for k in keys_of(i)) - 1

    def sort_key(i):
        density = amount[i] / costs[i] if costs[i] else amount[i]
        return (-ring_score[i], plaus[i], -density, i)

    order = sorted(range(N), key=sort_key)

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
