# TIER: strong
"""Model-based OPC: exact incremental coordinate descent that simulates the
KNOWN optical model (same kernel + doses the checker uses) and greedily
flips whichever single mask pixel most improves the process-window fidelity
objective F = 0.4*mean(IoU) + 0.6*min(IoU) jointly across ALL three doses.

This is the genuine insight over `greedy.py`: instead of one global bias
(or a correction tuned to a single dose), it directly inverts the
deterministic blur/threshold model per pixel, so isolated features get
grown, tightly pitched features get thinned, and corners get serifed --
each region receiving whatever local correction the model says it needs,
while every accepted move is verified (via exact incremental bookkeeping
of the intersection/union counts) to actually raise the worst-dose-aware
fidelity, not just the average.  Fully deterministic: fixed sweep order,
no randomness, no wall-clock dependence.
"""
import sys

A = [1, 2, 3, 2, 1]
DOSES = (33, 41, 49)
MEAN_WEIGHT = 0.4
MAX_SWEEPS = 10


def read_target():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = data[1:1 + n]
    target = [[1 if c == "1" else 0 for c in row] for row in rows]
    return n, target


def intensity_grid(mask, n):
    inten = [[0] * n for _ in range(n)]
    for x in range(n):
        for dx in range(-2, 3):
            xx = x + dx
            if xx < 0 or xx >= n:
                continue
            wx = A[dx + 2]
            row = mask[xx]
            for y in range(n):
                s = 0
                lo = max(0, y - 2)
                hi = min(n - 1, y + 2)
                for yy in range(lo, hi + 1):
                    s += row[yy] * A[yy - y + 2]
                inten[x][y] += s * wx
    return inten


def neighbors(x, y, n):
    out = []
    for dx in range(-2, 3):
        xx = x + dx
        if xx < 0 or xx >= n:
            continue
        wx = A[dx + 2]
        for dy in range(-2, 3):
            yy = y + dy
            if 0 <= yy < n:
                out.append((xx, yy, wx * A[dy + 2]))
    return out


def f_of(inter, uni):
    nd = len(inter)
    ious = [inter[d] / uni[d] if uni[d] > 0 else 1.0 for d in range(nd)]
    return MEAN_WEIGHT * (sum(ious) / nd) + (1.0 - MEAN_WEIGHT) * min(ious)


def optimize(target, n):
    nd = len(DOSES)
    mask = [row[:] for row in target]
    inten = intensity_grid(mask, n)
    printedv = [[[1 if inten[x][y] >= DOSES[d] else 0 for d in range(nd)]
                 for y in range(n)] for x in range(n)]
    inter = [0] * nd
    uni = [0] * nd
    for x in range(n):
        for y in range(n):
            t = target[x][y]
            for d in range(nd):
                p = printedv[x][y][d]
                if p and t:
                    inter[d] += 1
                if p or t:
                    uni[d] += 1

    cur_f = f_of(inter, uni)
    changed = True
    sweep = 0
    while changed and sweep < MAX_SWEEPS:
        changed = False
        sweep += 1
        for x in range(n):
            for y in range(n):
                nbrs = neighbors(x, y, n)
                delta = 1 - 2 * mask[x][y]
                new_bits = []
                for (xx, yy, w) in nbrs:
                    new_i = inten[xx][yy] + delta * w
                    old_p = printedv[xx][yy]
                    new_p = [1 if new_i >= DOSES[d] else 0 for d in range(nd)]
                    new_bits.append((xx, yy, new_i, old_p, new_p))

                d_inter = [0] * nd
                d_uni = [0] * nd
                for (xx, yy, new_i, old_p, new_p) in new_bits:
                    t = target[xx][yy]
                    for d in range(nd):
                        op, npd = old_p[d], new_p[d]
                        if op == npd:
                            continue
                        old_in = 1 if (op and t) else 0
                        new_in = 1 if (npd and t) else 0
                        old_un = 1 if (op or t) else 0
                        new_un = 1 if (npd or t) else 0
                        d_inter[d] += new_in - old_in
                        d_uni[d] += new_un - old_un

                trial_inter = [inter[d] + d_inter[d] for d in range(nd)]
                trial_uni = [uni[d] + d_uni[d] for d in range(nd)]
                trial_f = f_of(trial_inter, trial_uni)

                if trial_f > cur_f + 1e-12:
                    for (xx, yy, new_i, old_p, new_p) in new_bits:
                        inten[xx][yy] = new_i
                        printedv[xx][yy] = new_p
                    inter, uni = trial_inter, trial_uni
                    mask[x][y] = 1 - mask[x][y]
                    cur_f = trial_f
                    changed = True
    return mask


def main():
    n, target = read_target()
    mask = optimize(target, n)
    out = ["".join(str(c) for c in row) for row in mask]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
