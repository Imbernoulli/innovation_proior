# TIER: greedy
# The obvious first pass. Two textbook moves, no structural insight:
#  1) apply the weight-floor rule (w_i = 2^-size_i) to EVERY density level
#     -- this recovers the whole "safe prefix" where it happens to work.
#  2) where that fails, ESTIMATE the winner by a few seeded self-play
#     trajectories (both sides greedily grab whichever empty cell currently
#     sits on the most still-open lines) and, if Claimer looked good, ship a
#     "strategy table" built only from the states actually visited in those
#     few games.
# That table is a sample of a handful of trajectories, not a decision
# defined on every possible reply -- against most of the opponent's real
# branches it has no entry at all, so the checker's exhaustive replay finds
# a missing state and rejects it. No certificate, no credit, however
# confident the simulated win rate looked.
import sys
from fractions import Fraction

GREEDY_TRIALS = 3


def floor_weight_ok(line):
    return Fraction(1, 2 ** len(line))


def simulate(N, lines, trial_idx):
    """One greedy self-play trace; returns list of (hist, move) at claimer
    turns, and whether claimer completed a line."""
    line_sets = [frozenset(l) for l in lines]
    claimer, blocker = set(), set()
    hist = []
    records = []
    turn = 0
    while len(claimer) + len(blocker) < N:
        empties = [c for c in range(1, N + 1) if c not in claimer and c not in blocker]
        if not empties:
            break
        current = claimer if turn % 2 == 0 else blocker
        scores = {}
        for cell in empties:
            sc = 0
            for ls in line_sets:
                if cell in ls and not (ls & (blocker if turn % 2 == 0 else claimer)):
                    sc += 1
            scores[cell] = sc
        best = max(scores.values())
        cands = [c for c in empties if scores[c] == best]
        pick = cands[(trial_idx + turn) % len(cands)]
        if turn % 2 == 0:
            records.append((tuple(hist), pick))
            claimer.add(pick)
        else:
            blocker.add(pick)
        hist.append(pick)
        turn += 1
        if any(ls <= claimer for ls in line_sets):
            return records, True
    return records, any(ls <= claimer for ls in line_sets)


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    pool = []
    for _ in range(P):
        sz = int(next(it))
        cells = [int(next(it)) for _ in range(sz)]
        pool.append(cells)
    c = [int(next(it)) for _ in range(K)]

    out = [str(K)]
    for j in range(1, K + 1):
        lines = pool[:c[j - 1]]
        total = sum(floor_weight_ok(l) for l in lines)
        if total < Fraction(1, 2):
            weights = [f"1/{2 ** len(l)}" for l in lines]
            out.append(f"{j} B " + " ".join(weights))
            continue

        table = {}
        won_any = False
        for t in range(GREEDY_TRIALS):
            records, won = simulate(N, lines, t)
            if won:
                won_any = True
                for state, mv in records:
                    table.setdefault(state, mv)
        if won_any and table:
            rows = []
            for state, mv in table.items():
                rows.append(f"{len(state)} " + " ".join(str(x) for x in state) + f" {mv}")
            out.append(f"{j} M {len(rows)}")
            out.extend(rows)
        else:
            out.append(f"{j} U")
    print("\n".join(out))


if __name__ == "__main__":
    main()
