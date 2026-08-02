# TIER: greedy
"""The obvious recipe: incremental hill-climbing from the empty composition.

At each step, look at every element and ask "if I add one more unit of this
element, does the resulting (X, IM) still satisfy the budget of whatever band
X now falls into?". Among the elements for which the answer is yes, add a
unit to whichever gives the best strength-gain-per-brittleness-cost. Stop
when no element can be added without breaching its target band's budget.

This never produces an infeasible composition, and locally it is exactly the
right move at every step -- but it only ever walks forward from zero along a
single path. If a narrow, tight band lies between the current position and a
much more generous region further out, this walk freezes at the near edge of
the tight band and never crosses it, because crossing it would (at that
moment, along that path) require violating the tight band it has to pass
through first. It has no notion of jumping straight to a different regime.
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

    while True:
        best_i = -1
        best_ratio = -1.0
        newX = X + 1
        if newX > MAXX:
            break
        newBin = newX // W
        cap = T[newBin]
        for i in range(K):
            newIM = IM + b[i]
            if newIM > cap:
                continue
            gain = s[i] * (math.sqrt(x[i] + 1) - math.sqrt(x[i]))
            ratio = gain / b[i]
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        if best_i == -1:
            break
        x[best_i] += 1
        X += 1
        IM += b[best_i]

    print(' '.join(map(str, x)))


if __name__ == "__main__":
    main()
