# TIER: strong
"""The insight: the checker only ever looks at the FINAL submitted composition
-- feasibility depends solely on which band that composition's own total X
lands in, never on any path used to "reach" it. So a band that looks brittle
partway through building a composition is not actually disqualifying, as long
as it is not where you choose to stop.

Run the SAME cost-ratio marginal-exchange construction greedy.py uses --
repeatedly grant the next unit of solute to whichever element currently gives
the best strength-gain per unit of brittleness cost -- but do not stop at the
first band whose budget it would blow. Keep extending the construction all
the way to the total-solute cap, and simply remember the best FEASIBLE
composition seen at ANY point along the way (checking each point's own band
threshold on the spot). A band that pinches shut early no longer matters once
a later, wider band reopens, because nothing forces the submission to have
walked there through the pinch feasibly -- only the final artifact is graded.

Why this single monotone construction already contains the true optimum for
every possible brittleness budget: this is the classical marginal/exchange
allocation result for maximizing a sum of separable concave, non-decreasing
rewards (s_i*sqrt(x_i)) under one linear cost (b_i per unit), with FIXED
per-unit costs. Because each element's own marginal gain is non-increasing as
its own amount grows, always taking the globally best available gain/cost
ratio next is optimal for every budget prefix simultaneously (a standard
pairwise-exchange argument: swapping a lower-ratio increment already taken
for a higher-ratio one not yet taken, at equal cost, cannot decrease the
total). So scanning this one construction and keeping the best point whose
OWN band happens to accommodate it already finds the true optimum -- no
per-band restart or lookahead is needed, only refusing to stop early.
"""
import math
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    K = int(toks[idx]); idx += 1
    W = int(toks[idx]); idx += 1
    numBins = int(toks[idx]); idx += 1
    s = [int(toks[idx + i]) for i in range(K)]; idx += K
    b = [int(toks[idx + i]) for i in range(K)]; idx += K
    T = [int(toks[idx + i]) for i in range(numBins)]; idx += numBins
    MAXX = numBins * W - 1

    x = [0] * K
    X = 0
    IM = 0

    best_val = -1.0
    best_x = [0] * K

    for _ in range(MAXX):
        best_i = -1
        best_ratio = -1.0
        for i in range(K):
            gain = s[i] * (math.sqrt(x[i] + 1) - math.sqrt(x[i]))
            ratio = gain / b[i]
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        x[best_i] += 1
        X += 1
        IM += b[best_i]

        binIdx = X // W
        if IM <= T[binIdx]:
            val = sum(s[i] * math.sqrt(x[i]) for i in range(K))
            if val > best_val:
                best_val = val
                best_x = x[:]

    print(' '.join(map(str, best_x)))


if __name__ == "__main__":
    main()
