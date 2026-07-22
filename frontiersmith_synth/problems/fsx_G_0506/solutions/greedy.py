# TIER: greedy
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
            vs = tuple(sorted(int(next(it)) for _ in range(d)))
            terms.append((c, vs))
        targets.append(terms)
    return n, m, targets


def main():
    _, _, targets = read_instance()
    lines = []
    mono_cache = {}

    def new(op, a, b):
        name = "t%d" % len(lines)
        lines.append("%s %s %s %s" % (name, op, a, b))
        return name

    def monomial(vs):
        key = tuple(vs)
        if key in mono_cache:
            return mono_cache[key]
        if not key:
            op = "#1"
        else:
            cur = "x%d" % key[0]
            for v in key[1:]:
                cur = new("MUL", cur, "x%d" % v)
            op = cur
        mono_cache[key] = op
        return op

    def scale(c, op):
        sign = 1 if c > 0 else -1
        mag = abs(c)
        if mag != 1:
            op = new("MUL", "#%d" % mag, op)
        return sign, op

    def sum_signed(items):
        if not items:
            return "#0"
        first = next((i for i, (sgn, _) in enumerate(items) if sgn > 0), 0)
        sgn, acc = items[first]
        if sgn < 0:
            acc = new("SUB", "#0", acc)
        for i, (sgn, val) in enumerate(items):
            if i == first:
                continue
            acc = new("ADD" if sgn > 0 else "SUB", acc, val)
        return acc

    outs = []
    for terms in targets:
        items = [scale(c, monomial(vs)) for c, vs in terms]
        outs.append(sum_signed(items))

    sys.stdout.write("%d\n" % len(lines))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write("OUT " + " ".join(outs) + "\n")


if __name__ == "__main__":
    main()
