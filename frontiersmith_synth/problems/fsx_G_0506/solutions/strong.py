# TIER: strong
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
    prefix_cache = {}

    def new(op, a, b):
        name = "t%d" % len(lines)
        lines.append("%s %s %s %s" % (name, op, a, b))
        return name

    def monomial(vs):
        key = tuple(vs)
        if key in prefix_cache:
            return prefix_cache[key]
        if not key:
            return "#1"
        cur = None
        pref = ()
        for v in key:
            pref = pref + (v,)
            if pref in prefix_cache:
                cur = prefix_cache[pref]
                continue
            if cur is None:
                cur = "x%d" % v
            else:
                cur = new("MUL", cur, "x%d" % v)
            prefix_cache[pref] = cur
        return cur

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

    def find_groups(terms):
        cand = {}
        for idx, (c, vs) in enumerate(terms):
            if len(vs) < 3:
                continue
            for pos, tail in enumerate(vs):
                core = vs[:pos] + vs[pos + 1:]
                if len(core) < 2:
                    continue
                cand.setdefault(core, {})[idx] = (tail, c)

        scored = []
        for core, by_idx in cand.items():
            if len(by_idx) < 3:
                continue
            # Approximate weighted gain versus building each core*tail monomial.
            saving = len(by_idx) * (3 * len(core)) - (3 * max(0, len(core) - 1) + 3)
            scored.append((saving, len(by_idx), len(core), core, by_idx))
        scored.sort(reverse=True)

        used = set()
        groups = []
        for _, _, _, core, by_idx in scored:
            avail = [(idx, by_idx[idx][0], by_idx[idx][1]) for idx in sorted(by_idx) if idx not in used]
            if len(avail) < 3:
                continue
            groups.append((core, avail))
            for idx, _, _ in avail:
                used.add(idx)
        return groups, used

    outs = []
    for terms in targets:
        groups, used = find_groups(terms)
        pieces = []

        for core, entries in groups:
            core_op = monomial(core)
            tail_items = []
            for _, tail, coeff in entries:
                tail_items.append(scale(coeff, "x%d" % tail))
            combo = sum_signed(tail_items)
            pieces.append((1, new("MUL", core_op, combo)))

        for idx, (coeff, vs) in enumerate(terms):
            if idx in used:
                continue
            pieces.append(scale(coeff, monomial(vs)))

        outs.append(sum_signed(pieces))

    sys.stdout.write("%d\n" % len(lines))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write("OUT " + " ".join(outs) + "\n")


if __name__ == "__main__":
    main()
