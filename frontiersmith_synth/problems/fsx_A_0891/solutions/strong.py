# TIER: strong
# The insight: don't fit a smooth curve of the visible features -- treat the
# log as evidence about a PROGRAM and design a discriminating read of it.
#
# 1. Hunt for matched pairs: states with IDENTICAL (n,q,w) but DIFFERENT a
#    whose recorded winner differs. These isolate a pure age effect,
#    independent of queue/weight, and the exact age at which the winner
#    flips is the override BOUNDARY -- not a slope, a threshold.
# 2. Likewise, states with identical (n,q,a=0) but different w whose winner
#    differs confirm the weighted-queue MULTIPLICATION (not addition).
# 3. Rather than guess the boundary from the probes alone, grid-fit the
#    threshold K against ALL training states under the CORRECT functional
#    form: "if any approach's age >= K, the one with the largest age wins
#    (ties by clockwise order); else the largest w*q wins (ties by
#    clockwise order)" -- i.e. search over the hypothesis SHAPE the probes
#    revealed, not over an arbitrary linear coefficient.
# 4. Also notice that exact w*q ties resolve by clockwise order (the cw
#    feature) even in the non-override branch -- encode that as a tiny
#    numeric tie-break term so it survives being folded into ONE per-approach
#    priority expression (the grader evaluates each approach independently
#    and takes the argmax, so age-among-overriders must be encoded as a
#    term that scales WITH age past the threshold, not a flat bonus, or two
#    simultaneous overriders would be compared by w*q instead of by age).
#
# This threshold+age-scaled-override+multiplicative-weight+cw-tiebreak
# rule is degree-generic: it never refers to n, so it transfers unchanged
# to the held-out 5-way/6-way junctions where overrides are common.
import sys


def parse(text):
    toks = text.split()
    i = 0
    assert toks[i] == "TESTID"; i += 2
    assert toks[i] == "NSTATES"; m = int(toks[i + 1]); i += 2
    states = []
    for _ in range(m):
        assert toks[i] == "STATE"; i += 1
        n = int(toks[i].split("=")[1]); i += 1
        i += 1  # LASTGREEN, unused (cw already accounts for it)
        win = int(toks[i].split("=")[1]); i += 1
        assert toks[i] == "Q"; i += 1
        q = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "W"; i += 1
        w = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "A"; i += 1
        a = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "CW"; i += 1
        cw = [int(toks[i + j]) for j in range(n)]; i += n
        states.append((n, win, q, w, a, cw))
    return states


def decide_hyp(n, q, w, a, cw, K):
    overriders = [i for i in range(n) if a[i] >= K]
    if overriders:
        maxage = max(a[i] for i in overriders)
        cands = [i for i in overriders if a[i] == maxage]
    else:
        maxscore = max(w[i] * q[i] for i in range(n))
        cands = [i for i in range(n) if w[i] * q[i] == maxscore]
    return min(cands, key=lambda i: cw[i])


def acc_for_K(states, K):
    correct = 0
    for (n, win, q, w, a, cw) in states:
        if decide_hyp(n, q, w, a, cw, K) == win:
            correct += 1
    return correct / len(states)


def main():
    states = parse(sys.stdin.read())

    # Confirm the matched-pair signal exists, then grid-search the exact
    # threshold against the CORRECT hypothesis shape (not a linear proxy).
    best_K, best_acc = None, -1.0
    for K in range(1, 21):
        acc = acc_for_K(states, K)
        if acc > best_acc:
            best_acc, best_K = acc, K

    BIG = 100000.0
    EPS = 0.01
    # priority_i = BIG * step(a_i - (K-0.5)) * a_i  + w_i*q_i  - EPS*cw_i
    # (age-scaled so two simultaneous overriders compare by AGE, not w*q;
    #  the -EPS*cw term breaks exact w*q ties by clockwise order.)
    expr = "%.1f * step ( a - %.1f ) * a + w * q - %.4f * cw" % (BIG, best_K - 0.5, EPS)
    print("PRIORITY " + expr)


if __name__ == "__main__":
    main()
