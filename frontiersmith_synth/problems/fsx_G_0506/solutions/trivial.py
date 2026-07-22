# TIER: trivial
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    targets = []
    for _ in range(m):
        k = int(next(it))
        terms = []
        for _ in range(k):
            c = int(next(it))
            d = int(next(it))
            vs = tuple(int(next(it)) for _ in range(d))
            terms.append((c, vs))
        targets.append(terms)
    return n, m, targets


def main():
    _, _, targets = read_instance()
    lines = []

    def new(op, a, b):
        name = "t%d" % len(lines)
        lines.append("%s %s %s %s" % (name, op, a, b))
        return name

    def monomial(vs):
        if not vs:
            return "#1"
        cur = "x%d" % vs[0]
        for v in vs[1:]:
            cur = new("MUL", cur, "x%d" % v)
        return cur

    def scale(c, op):
        sign = 1 if c > 0 else -1
        mag = abs(c)
        if mag != 1:
            op = new("MUL", "#%d" % mag, op)
        return sign, op

    outs = []
    for terms in targets:
        signed = [scale(c, monomial(vs)) for c, vs in terms]
        first = next((i for i, (sgn, _) in enumerate(signed) if sgn > 0), 0)
        sgn, acc = signed[first]
        if sgn < 0:
            acc = new("SUB", "#0", acc)
        for i, (sgn, val) in enumerate(signed):
            if i == first:
                continue
            acc = new("ADD" if sgn > 0 else "SUB", acc, val)
        outs.append(acc)

    sys.stdout.write("%d\n" % len(lines))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write("OUT " + " ".join(outs) + "\n")


if __name__ == "__main__":
    main()
