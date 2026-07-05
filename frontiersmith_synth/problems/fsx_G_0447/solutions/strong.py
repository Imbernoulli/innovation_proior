# TIER: strong
# CSE with SUB-PRODUCT (prefix) sharing: canonicalize each monomial by sorting
# its variables, then build products through a shared prefix trie so common
# leading sub-products are computed only once across ALL monomials.
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

    prefix_op = {}  # sorted-prefix tuple -> operand

    def get_mono(vs):
        key = tuple(sorted(vs))
        cur = None
        pref = ()
        for v in key:
            pref = pref + (v,)
            if pref in prefix_op:
                cur = prefix_op[pref]
            else:
                if cur is None:
                    cur = "x%d" % v          # length-1 prefix: a bare variable
                else:
                    cur = new("MUL", cur, "x%d" % v)
                prefix_op[pref] = cur
        return cur

    outs = []
    for terms in targets:
        term_vals = []
        for (c, vs) in terms:
            prod = get_mono(vs)
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
