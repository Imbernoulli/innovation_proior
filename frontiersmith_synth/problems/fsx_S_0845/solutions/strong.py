# TIER: strong
# Insight: a bloc can only steer the outcome if the rule gives it a stable target to push
# toward. Two-step defense:
#  1) Restrict to the Smith set (the smallest set of candidates that pairwise-majority-beats
#     every candidate outside it). This is an INVARIANT of the majority relation: as long as a
#     b-ballot rewrite doesn't flip enough individual pairwise majorities to change who is IN
#     the Smith set, the winner can only come from that set -- a much smaller target than "any
#     of 10 candidates" a plurality/Borda-style rule exposes.
#  2) Within the Smith set (only nonempty when a cycle survives -- e.g. after a CYCLE-INJECT
#     bloc), break ties by MINIMAX margin (the candidate with the smallest worst pairwise
#     defeat), not by raw candidate index. A fixed index-based tie-break gives the bloc a
#     predictable, engineerable target (whichever cycle member has the lowest index); minimax
#     ties the outcome to the actual vote margins instead. Any residual exact tie is broken by
#     a hash of the full ballot multiset -- a pure function of the exact votes cast, so it
#     offers the bloc no fixed candidate to aim for independent of the votes they themselves
#     cast.
import sys
import json
import hashlib


def pairwise(ballots, m):
    W = [[0] * m for _ in range(m)]
    for bal in ballots:
        pos = {c: i for i, c in enumerate(bal)}
        for i in range(m):
            for j in range(i + 1, m):
                if pos[i] < pos[j]:
                    W[i][j] += 1
                else:
                    W[j][i] += 1
    return W


def smith_set(W, m):
    beats = [[W[i][j] > W[j][i] for j in range(m)] for i in range(m)]
    reach = [row[:] for row in beats]
    for k in range(m):
        for i in range(m):
            if reach[i][k]:
                for j in range(m):
                    if reach[k][j]:
                        reach[i][j] = True
    return [i for i in range(m) if all(not (reach[j][i] and not reach[i][j]) for j in range(m))]


def main():
    inst = json.load(sys.stdin)
    ballots = inst["ballots"]
    m = inst["num_candidates"]

    W = pairwise(ballots, m)
    S = smith_set(W, m)

    if len(S) == 1:
        winner = S[0]
    else:
        def worst_margin(c):
            return max((W[j][c] - W[c][j] for j in S if j != c), default=0)

        best = min(worst_margin(c) for c in S)
        finalists = [c for c in S if worst_margin(c) == best]
        if len(finalists) == 1:
            winner = finalists[0]
        else:
            def key(c):
                h = hashlib.sha256()
                for bal in ballots:
                    h.update(bytes(bal))
                h.update(bytes([c]))
                return h.hexdigest()

            winner = min(finalists, key=key)

    print(json.dumps({"winner": winner}))


if __name__ == "__main__":
    main()
