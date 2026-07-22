# TIER: strong
# The insight: with non-consecutive snapshots you cannot fit one-step
# transitions directly. But the RULE SPACE is finite (256 possible radius-1
# tables). So instead of regressing transitions, INVERT the k-fold
# composition by constraint satisfaction: for every candidate rule, roll it
# forward from each observed snapshot by EXACTLY the recorded tick gap to
# the next observed snapshot, and score the candidate by total mismatch
# against what was actually observed there (noise-tolerant, not exact-match).
# The candidate consistent with ALL gaps simultaneously is (almost always)
# the true rule -- this is a search over a finite space, not a per-cell fit.
import sys


def step(row, table, W):
    return [table[row[(i - 1) % W] * 4 + row[i] * 2 + row[(i + 1) % W]] for i in range(W)]


def roll(row, table, W, steps):
    for _ in range(steps):
        row = step(row, table, W)
    return row


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    times, rows = [], []
    for _ in range(m):
        ti = int(data[idx]); idx += 1
        row_s = data[idx]; idx += 1
        times.append(ti)
        rows.append([int(c) for c in row_s])

    best_mismatch = None
    best_rule = 0
    for rulenum in range(256):
        table = [(rulenum >> k) & 1 for k in range(8)]
        total = 0
        cnt = 0
        for a in range(m - 1):
            gap = times[a + 1] - times[a]
            pred = roll(rows[a], table, W, gap)
            tgt = rows[a + 1]
            total += sum(1 for x, y in zip(pred, tgt) if x != y)
            cnt += W
        mismatch = total / float(cnt)
        if best_mismatch is None or mismatch < best_mismatch:
            best_mismatch = mismatch
            best_rule = rulenum

    table = [(best_rule >> k) & 1 for k in range(8)]
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
