# TIER: greedy
# The obvious recipe: pool every (earlier-snapshot 3-window -> later-snapshot
# cell) example across ALL consecutive LOGBOOK pairs as if the gap between
# them were a single tick, and majority-vote a radius-1 rule table from the
# pooled counts. More data looks like it should mean a better fit -- but a
# radius-1 window seen k>=2 ticks before the target cell does NOT determine
# that cell under the true rule (the true dependency has widened to radius
# k), so for every gappy pair the same window maps to inconsistent outcomes
# depending on cells outside the window, and majority vote just absorbs that
# as noise. This never looks at the actual recorded tick numbers at all.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    rows = []
    for _ in range(m):
        idx += 1  # tick number, unused (that's the trap)
        row_s = data[idx]; idx += 1
        rows.append([int(c) for c in row_s])

    counts = [[0, 0] for _ in range(8)]
    for a in range(m - 1):
        row_prev, row_next = rows[a], rows[a + 1]
        for i in range(W):
            cl = row_prev[(i - 1) % W]
            cm = row_prev[i]
            cr = row_prev[(i + 1) % W]
            k = cl * 4 + cm * 2 + cr
            counts[k][row_next[i]] += 1

    # ties (no clear majority evidence) default to "no change" rather than an
    # arbitrary coin flip -- still the obvious recipe, just not a reckless one.
    table = []
    for k in range(8):
        c0, c1 = counts[k]
        if c1 > c0:
            table.append(1)
        elif c0 > c1:
            table.append(0)
        else:
            table.append((k >> 1) & 1)

    ones = [k for k in range(8) if table[k]]
    if not ones:
        print("0")
    elif len(ones) == 8:
        print("1")
    else:
        clauses = []
        for k in ones:
            cl, cm, cr = (k >> 2) & 1, (k >> 1) & 1, k & 1
            clauses.append("( cL == %d and cM == %d and cR == %d )" % (cl, cm, cr))
        print(" or ".join(clauses))


if __name__ == "__main__":
    main()
