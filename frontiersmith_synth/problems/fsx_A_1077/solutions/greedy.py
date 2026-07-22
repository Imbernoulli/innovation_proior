# TIER: greedy
# The "obvious" common-subexpression elimination: build each output row as a
# single chain over its nonzero terms (leading with a free +1-sign input
# reference when available, exactly like the tight per-row baseline), but
# memoize on the EXACT PREFIX of (id, sign) pairs seen so far so that two
# rows sharing an identical INITIAL run of terms reuse the same temporaries.
# This is a real, valid single-pass CSE a competent coder writes first -- but
# a linear chain can only ever share a PREFIX with another row; it cannot
# notice that two rows share an interior (non-prefix) partial sum, which is
# exactly where the planted structure hides its reuse.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [[int(next(it)) for _ in range(n)] for _ in range(n)]

    instrs = []
    next_id = [n + 1]
    cache = {(): 0}  # empty prefix -> constant zero (free)
    outs = []

    for i in range(n):
        terms = [(j + 1, A[i][j]) for j in range(n) if A[i][j] != 0]
        for (_, v) in terms:
            assert v == 1 or v == -1
        pos = [t for t in terms if t[1] == 1]
        neg = [t for t in terms if t[1] == -1]
        ordered = pos + neg if pos else neg

        prefix = ()
        acc = 0
        first = True
        for (val_id, sign) in ordered:
            newprefix = prefix + ((val_id, sign),)
            if newprefix in cache:
                acc = cache[newprefix]
                prefix = newprefix
                first = False
                continue
            if first and sign == 1:
                newid = val_id  # free direct input reference, no gate
            elif first:
                newid = next_id[0]
                next_id[0] += 1
                instrs.append((newid, 0, "-", val_id))
            else:
                op = "+" if sign == 1 else "-"
                newid = next_id[0]
                next_id[0] += 1
                instrs.append((newid, acc, op, val_id))
            cache[newprefix] = newid
            acc = newid
            prefix = newprefix
            first = False
        outs.append(acc)

    out = [str(len(instrs))]
    for (idx, a, op, b) in instrs:
        out.append("%d %d %s %d" % (idx, a, op, b))
    out.append(" ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
