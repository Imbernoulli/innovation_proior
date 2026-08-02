# TIER: greedy
import sys


def compute_borda(ballots, m):
    scores = [0] * m
    for ballot in ballots:
        for pos, cand in enumerate(ballot):
            scores[cand] += (m - 1 - pos)
    return scores


def score_rule_winner(ballots, w, m):
    totals = [0] * m
    for ballot in ballots:
        for pos, cand in enumerate(ballot):
            totals[cand] += w[pos]
    best = 0
    for c in range(1, m):
        if totals[c] > totals[best]:
            best = c
    return best


def runoff2_winner(ballots, m):
    scores = compute_borda(ballots, m)
    order = sorted(range(m), key=lambda c: (-scores[c], c))
    f1, f2 = order[0], order[1]
    v1 = v2 = 0
    for ballot in ballots:
        if ballot.index(f1) < ballot.index(f2):
            v1 += 1
        else:
            v2 += 1
    if v1 > v2:
        return f1
    elif v2 > v1:
        return f2
    return min(f1, f2)


def simulate(ballots, m, rule_type, weights):
    if rule_type == 0:
        return score_rule_winner(ballots, weights, m)
    return runoff2_winner(ballots, m)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it)); rule_type = int(next(it)); target = int(next(it))
    weights = None
    if rule_type == 0:
        weights = [int(next(it)) for _ in range(m)]
    ballots = [[int(next(it)) for _ in range(m)] for _ in range(n)]

    # The "obvious" single-pass recipe: whoever currently wins is treated as THE
    # threat. Coalition voters are converted, in input order, to "target first,
    # current winner last, everyone else keeps their original relative order" --
    # the textbook scoring-rule manipulation trick -- and we stop as soon as the
    # rule flips to target. No search over which rival to bury, no reordering of
    # who joins the coalition first.
    rival = simulate(ballots, m, rule_type, weights)

    coalition_assignment = None
    for k in range(1, n + 1):
        modified = []
        for i, orig in enumerate(ballots):
            if i < k:
                rest = [c for c in orig if c != target and c != rival]
                modified.append([target] + rest + [rival])
            else:
                modified.append(orig)
        w = simulate(modified, m, rule_type, weights)
        if w == target:
            coalition_assignment = modified[:k]
            break

    if coalition_assignment is None:
        # Should not happen (full coalition always works) -- fall back to k=n.
        k = n
        coalition_assignment = []
        for i, orig in enumerate(ballots):
            rest = [c for c in orig if c != target and c != rival]
            coalition_assignment.append([target] + rest + [rival])

    lines = [str(k)]
    for i in range(k):
        lines.append(str(i) + " " + " ".join(map(str, coalition_assignment[i])))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
