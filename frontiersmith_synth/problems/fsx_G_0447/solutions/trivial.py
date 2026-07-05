# TIER: trivial
# Naive per-term construction: rebuild every monomial from scratch, no sharing.
# Reproduces the checker's internal baseline B -> ratio ~= 0.1.
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    targets = []
    for _ in range(m):
        K = int(next(it)); terms = []
        for _ in range(K):
            c = int(next(it)); d = int(next(it))
            vs = [int(next(it)) for _ in range(d)]
            terms.append((c, vs))
        targets.append(terms)
    return n, m, targets


def main():
    n, m, targets = read_instance()
    lines = []
    k = 0

    def new(op, a, b):
        nonlocal k
        name = "t%d" % k; k += 1
        lines.append("%s %s %s %s" % (name, op, a, b))
        return name

    outs = []
    for terms in targets:
        term_vals = []
        for (c, vs) in terms:
            if len(vs) == 1:
                prod = "x%d" % vs[0]
            else:
                cur = "x%d" % vs[0]
                for v in vs[1:]:
                    cur = new("MUL", cur, "x%d" % v)
                prod = cur
            if abs(c) != 1:
                term_vals.append((new("MUL", "#%d" % c, prod), 1))
            else:
                term_vals.append((prod, 1 if c > 0 else -1))
        acc = term_vals[0][0]
        for (val, sign) in term_vals[1:]:
            acc = new("ADD" if sign > 0 else "SUB", acc, val)
        outs.append(acc)

    sys.stdout.write("%d\n" % len(lines))
    sys.stdout.write("\n".join(lines))
    if lines:
        sys.stdout.write("\n")
    sys.stdout.write("OUT " + " ".join(outs) + "\n")


if __name__ == "__main__":
    main()
