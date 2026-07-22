# TIER: strong
# Common-subexpression discovery via RECURSIVE best-pair merging (a
# Boyar-Peralta-style "new distinct base" heuristic), the real insight the
# problem is built around: repeatedly find the (id_a, sign_a, id_b, sign_b)
# signed pair that currently occurs, verbatim, inside the most not-yet-
# finished rows -- where id_a/id_b may themselves be PREVIOUSLY DISCOVERED
# shared atoms, not just raw inputs -- turn it into ONE new shared
# temporary, and replace that pair everywhere it appears. Because merged
# atoms re-enter the same pool, this recovers MULTI-LEVEL sharing (atoms
# built out of atoms, mirroring a hidden butterfly-of-butterflies), which a
# single left-to-right pass over each row in isolation cannot see: a chain
# can only ever notice a coincidental shared PREFIX, never that two rows
# share an interior partial-sum-of-partial-sums.
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [[int(next(it)) for _ in range(n)] for _ in range(n)]

    # cur[r]: current list of (id, sign) terms still needing to be combined
    # to reconstruct row r. Starts as the row's raw (input_id, coefficient)
    # pairs; shrinks by one entry every time two of its entries are folded
    # into a single shared atom.
    cur = []
    for r in range(n):
        terms = [(j + 1, A[r][j]) for j in range(n) if A[r][j] != 0]
        for (_, v) in terms:
            assert v == 1 or v == -1
        cur.append(terms)

    instrs = []
    next_id = [n + 1]

    while True:
        counts = defaultdict(int)
        rows_for = defaultdict(list)
        for r in range(n):
            lst = cur[r]
            m = len(lst)
            for a in range(m):
                ida, sa = lst[a]
                for b in range(a + 1, m):
                    idb, sb = lst[b]
                    lo, hi = (ida, idb) if ida < idb else (idb, ida)
                    slo = sa if ida < idb else sb
                    shi = sb if ida < idb else sa
                    key = (lo, hi, slo * shi)
                    counts[key] += 1
                    rows_for[key].append(r)
        if not counts:
            break
        # deterministic best pick: max count, then lexicographically smallest key
        best_key = min(counts.keys(), key=lambda k: (-counts[k], k))
        if counts[best_key] < 2:
            break
        lo, hi, srel = best_key
        idx = next_id[0]
        next_id[0] += 1
        op = "+" if srel == 1 else "-"
        instrs.append((idx, lo, op, hi))
        for r in rows_for[best_key]:
            lst = cur[r]
            i_lo = next((i for i, (v, _) in enumerate(lst) if v == lo), None)
            i_hi = next((i for i, (v, _) in enumerate(lst) if v == hi), None)
            if i_lo is None or i_hi is None:
                continue  # already consumed by an earlier pick this round
            slo, shi = lst[i_lo][1], lst[i_hi][1]
            if slo * shi != srel:
                continue  # residual changed since counting; skip (stays correct)
            for i in sorted((i_lo, i_hi), reverse=True):
                lst.pop(i)
            lst.append((idx, slo))

    # final per-row assembly: chain the remaining parts together, ordering a
    # '+'-sign part first when available so the leading step needs no
    # constant-zero gate
    outs = []
    for r in range(n):
        plist = cur[r]
        pos = [t for t in plist if t[1] == 1]
        neg = [t for t in plist if t[1] == -1]
        ordered = (pos + neg) if pos else neg
        if not ordered:
            outs.append(0)
            continue
        first_id, first_sign = ordered[0]
        if first_sign == 1:
            acc = first_id
        else:
            idx = next_id[0]
            next_id[0] += 1
            instrs.append((idx, 0, "-", first_id))
            acc = idx
        for (val_id, sign) in ordered[1:]:
            op = "+" if sign == 1 else "-"
            idx = next_id[0]
            next_id[0] += 1
            instrs.append((idx, acc, op, val_id))
            acc = idx
        outs.append(acc)

    out = [str(len(instrs))]
    for (idx, a, op, b) in instrs:
        out.append("%d %d %s %d" % (idx, a, op, b))
    out.append(" ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
