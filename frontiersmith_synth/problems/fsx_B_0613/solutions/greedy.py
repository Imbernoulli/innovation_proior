# TIER: greedy
# The obvious recipe: drop in a good general-purpose sorting network (Batcher's
# odd-even mergesort, pruned to n wires). Fewer comparators than insertion sort,
# but it RE-SORTS inside the already-ordered runs -- it never looks at the planted
# partial order, so it wastes comparators re-verifying known relations.
import sys

def pow2ge(x):
    t = 1
    while t < x:
        t <<= 1
    return t

def oe_merge(lo, n2, r, n, comps):
    step = r * 2
    if step < n2:
        oe_merge(lo, n2, step, n, comps)
        oe_merge(lo + r, n2, step, n, comps)
        i = lo + r
        while i + r < lo + n2:
            if i < n and i + r < n:
                comps.append((i, i + r))
            i += step
    else:
        if lo < n and lo + r < n:
            comps.append((lo, lo + r))

def oe_sort(lo, n2, n, comps):
    if n2 > 1:
        m = n2 // 2
        oe_sort(lo, m, n, comps)
        oe_sort(lo + m, m, n, comps)
        oe_merge(lo, n2, 1, n, comps)

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    comps = []
    oe_sort(0, pow2ge(n), n, comps)
    out = [str(len(comps))]
    for (a, b) in comps:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
