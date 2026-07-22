# TIER: greedy
# The obvious recipe: the visible features are q, w, a -- so score each
# approach by its weighted queue w*q, plus a LINEAR bonus for age (a smooth
# "older waits get a boost", the natural single-pass instinct), and grid-fit
# the one free coefficient against the TRAINING log's own accuracy. No
# thresholding, no tie-break term.
#
# On the training junctions (3-way/4-way, where the starvation rule almost
# never fires -- an approach rarely waits long enough to matter) a small or
# zero age-coefficient already gets near-perfect training accuracy, so the
# grid search settles on a small nudge.  A LINEAR bonus can never reproduce a
# hard threshold: it either overrides too early (letting a merely-slightly-
# older low-queue approach steal cycles it shouldn't) or too late/never for
# genuinely starved approaches once a high-queue rival is present.  On the
# held-out 5-way/6-way junctions, where the same starvation rule fires far
# more often (more rivals per approach means longer typical waits), this
# mismatch is common, not rare.
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
        lg = int(toks[i].split("=")[1]); i += 1
        win = int(toks[i].split("=")[1]); i += 1
        assert toks[i] == "Q"; i += 1
        q = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "W"; i += 1
        w = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "A"; i += 1
        a = [int(toks[i + j]) for j in range(n)]; i += n
        assert toks[i] == "CW"; i += 1
        cw = [int(toks[i + j]) for j in range(n)]; i += n
        states.append((n, lg, win, q, w, a, cw))
    return states


def acc_for(states, c):
    correct = 0
    for (n, lg, win, q, w, a, cw) in states:
        vals = [w[i] * q[i] + c * a[i] for i in range(n)]
        best = max(range(n), key=lambda i: (vals[i], -i))
        if best == win:
            correct += 1
    return correct / len(states)


def main():
    states = parse(sys.stdin.read())
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    best_c, best_acc = 0.0, -1.0
    for c in grid:
        acc = acc_for(states, c)
        if acc > best_acc:
            best_acc, best_c = acc, c
    print("PRIORITY w * q + %.4f * a" % best_c)


if __name__ == "__main__":
    main()
