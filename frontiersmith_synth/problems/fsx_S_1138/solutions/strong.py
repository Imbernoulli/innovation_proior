# TIER: strong
# Insight: every PUNCH batches the orbit of the current fold subgroup, so the
# program really is a short chain of subgroups compressing the target set --
# and that chain should be DISCOVERED from the data, not assumed from visual
# (global) symmetry.  We interleave FOLD_X / FOLD_Y one level at a time.  At
# each level, for every mirror pair about to merge we check whether it is
# safe (both sides agree); if not, we PUNCH the True side right now, at the
# finest resolution where it is still distinguishable from its mismatched
# partner -- a "correction at an intermediate fold depth" -- and let folding
# continue regardless.  A single stray defect therefore costs a handful of
# local corrections instead of aborting the whole compression.  Finally we
# search over every prefix-depth actually reached and keep the ONE stopping
# point that minimizes total instruction count (folding further than the
# data supports only buys more corrections, not more batching).
import sys

TRUE, FALSE, DONE = 1, 0, -1


def fold_axis(val, Wc, Hc, axis):
    corrections = []
    if axis == 'X':
        newW = Wc // 2
        newval = [[None] * Hc for _ in range(newW)]
        for x in range(newW):
            row_a = val[x]; row_b = val[Wc - 1 - x]
            nr = newval[x]
            for y in range(Hc):
                a = row_a[y]; b = row_b[y]
                if a == b:
                    nr[y] = a
                else:
                    if a == TRUE:
                        corrections.append((x, y))
                    if b == TRUE:
                        corrections.append((Wc - 1 - x, y))
                    nr[y] = DONE
        return newval, newW, Hc, corrections
    else:
        newH = Hc // 2
        newval = [[None] * newH for _ in range(Wc)]
        for x in range(Wc):
            row = val[x]; nr = newval[x]
            for y in range(newH):
                a = row[y]; b = row[Hc - 1 - y]
                if a == b:
                    nr[y] = a
                else:
                    if a == TRUE:
                        corrections.append((x, y))
                    if b == TRUE:
                        corrections.append((x, Hc - 1 - y))
                    nr[y] = DONE
        return newval, Wc, newH, corrections


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it))
    val = [[FALSE] * N for _ in range(N)]
    for _ in range(H):
        i = int(next(it)); j = int(next(it))
        val[i][j] = TRUE

    k = N.bit_length() - 1  # N is a power of two

    cur = val
    cW, cH = N, N
    axis_seq = []
    corrections_seq = []
    snapshots = [(cW, cH, [row[:] for row in cur])]

    for _ in range(k):
        if cW >= 2 and cW % 2 == 0:
            nv, nW, nH, corr = fold_axis(cur, cW, cH, 'X')
            axis_seq.append('X'); corrections_seq.append(corr)
            cur, cW, cH = nv, nW, nH
            snapshots.append((cW, cH, [row[:] for row in cur]))
        if cH >= 2 and cH % 2 == 0:
            nv, nW, nH, corr = fold_axis(cur, cW, cH, 'Y')
            axis_seq.append('Y'); corrections_seq.append(corr)
            cur, cW, cH = nv, nW, nH
            snapshots.append((cW, cH, [row[:] for row in cur]))

    T = len(axis_seq)
    cum_ops = [0] * (T + 1)
    running = 0
    for t in range(1, T + 1):
        running += 1 + len(corrections_seq[t - 1])
        cum_ops[t] = running

    true_counts = []
    for (W, Ht, v) in snapshots:
        true_counts.append(sum(1 for row in v for cell in row if cell == TRUE))

    best_total, best_t = None, 0
    for t in range(T + 1):
        total = cum_ops[t] + true_counts[t] + 1
        if best_total is None or total < best_total:
            best_total, best_t = total, t

    ops = []
    for lvl in range(best_t):
        for (x, y) in corrections_seq[lvl]:
            ops.append("PUNCH %d %d" % (x, y))
        ops.append("FOLD_X" if axis_seq[lvl] == 'X' else "FOLD_Y")

    Wf, Hf, vf = snapshots[best_t]
    for x in range(Wf):
        row = vf[x]
        for y in range(Hf):
            if row[y] == TRUE:
                ops.append("PUNCH %d %d" % (x, y))
    ops.append("UNFOLD_ALL")
    sys.stdout.write("\n".join(ops) + "\n")


if __name__ == "__main__":
    main()
