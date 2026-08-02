# TIER: strong
import sys, itertools


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

    # Insight: the checker only ever asks "does the rule, RE-RUN on the modified
    # profile, elect target?" -- so the rule itself is the oracle. Instead of
    # guessing one fixed rewrite (target-first / current-winner-last) and one
    # fixed voter order, directly construct the minimal witness from the rule's
    # structure: try EVERY full reassignment template (which rival(s) get buried
    # in which slot -- this matters a lot for RUNOFF2, where burying the wrong
    # rival just installs a different, equally dangerous finalist) crossed with a
    # couple of principled voter-selection orders, and take the smallest coalition
    # that any of them certifies by direct re-simulation.
    others = [c for c in range(m) if c != target]

    order_index = list(range(n))
    order_worst_first = sorted(range(n), key=lambda i: ballots[i].index(target), reverse=True)
    voter_orders = [order_index, order_worst_first]

    best_k = None
    best_assignment = None

    for perm in itertools.permutations(others):
        template = [target] + list(perm)
        for order in voter_orders:
            limit = best_k if best_k is not None else n
            for k in range(1, limit + 1):
                flip = set(order[:k])
                modified = [template if i in flip else orig for i, orig in enumerate(ballots)]
                w = simulate(modified, m, rule_type, weights)
                if w == target:
                    if best_k is None or k < best_k:
                        best_k = k
                        best_assignment = [(i, template) for i in order[:k]]
                    break

    if best_k is None:
        # Guaranteed fallback (matches the checker's own baseline): full coalition.
        best_k = n
        best_assignment = [(i, [target] + others) for i in range(n)]

    lines = [str(best_k)]
    for i, ballot in best_assignment:
        lines.append(str(i) + " " + " ".join(map(str, ballot)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
