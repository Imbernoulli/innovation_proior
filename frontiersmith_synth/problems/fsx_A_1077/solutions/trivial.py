# TIER: trivial
# The tightest possible independent-per-row construction with ZERO cross-row
# sharing: for each output row, chain its nonzero terms into an accumulator,
# leading with a +1-sign term directly (its input id is already available for
# free) when one exists so no gate is wasted on a leading "0 op x"; only pay
# that extra gate when every nonzero entry in the row is negative. This is
# exactly the checker's baseline B -- no row ever looks at another row.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [[int(next(it)) for _ in range(n)] for _ in range(n)]

    instrs = []
    next_id = n + 1
    outs = []
    for i in range(n):
        terms = [(j + 1, A[i][j]) for j in range(n) if A[i][j] != 0]
        for (_, v) in terms:
            assert v == 1 or v == -1
        pos = [t for t in terms if t[1] == 1]
        neg = [t for t in terms if t[1] == -1]
        ordered = pos + neg if pos else neg

        if not ordered:
            outs.append(0)
            continue

        first_id, first_sign = ordered[0]
        if first_sign == 1:
            acc = first_id  # free direct reference, no gate spent
        else:
            instrs.append((next_id, 0, "-", first_id))
            acc = next_id
            next_id += 1
        for (val_id, sign) in ordered[1:]:
            op = "+" if sign == 1 else "-"
            instrs.append((next_id, acc, op, val_id))
            acc = next_id
            next_id += 1
        outs.append(acc)

    out = [str(len(instrs))]
    for (idx, a, op, b) in instrs:
        out.append("%d %d %s %d" % (idx, a, op, b))
    out.append(" ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
