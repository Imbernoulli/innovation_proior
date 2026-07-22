# TIER: strong
"""Joint (vectorial) addition-chain search: instead of chaining each target from
scratch, INVERT the op count into one shared build DAG.

Core method: a Bos-Coster-style reduction over the whole pending target set.
Repeatedly take the two largest pending values f1 >= f2 and rewrite f1 so that
its computation will reuse registers the other targets need anyway:
  - f1 = f2 + r        (subtract: r < f2 enters the pending set)
  - f1 = q*f2 + r      (approximate GCD: build the rail multiples 2*f2..q*f2 once,
                        then only r remains; this is what discovers the hidden
                        common rails t = R*u (+ c) -- every target collapses onto
                        shared multiples of one rail register)
  - f1 = h + (f1 - h)  (halve when f1 dwarfs f2, h = f1 // 2)
The reduction terminates at the free register 1; emitting the dependency DAG in
increasing order yields the shared chain.

Fallback: a recursive largest-difference construction (build t = g + d from the
largest existing register g < t, recursing on the difference). The shorter of the
two programs is emitted -- a portfolio of two genuinely different joint-chain
algorithms, both exploiting cross-target register reuse."""
import sys

sys.setrecursionlimit(1000000)

QCAP = 8  # max quotient for the approximate-GCD rail reduction


def coef_chain(q):
    """Binary addition chain of coefficients ending at q; each entry is the sum
    of two earlier entries."""
    bits = bin(q)[3:]
    seq = [1]
    cur = 1
    for b in bits:
        seq.append(cur + cur)
        cur += cur
        if b == "1":
            seq.append(cur + 1)
            cur += 1
    return seq


def boscoster_deps(targets):
    """Return dict deps[v] = (a, b) with v = a + b, covering all targets."""
    have = {1}
    deps = {}
    pend = set(targets) - have
    while pend:
        L = sorted(pend, reverse=True)
        f1 = L[0]
        f2 = L[1] if len(L) > 1 else 0
        pend.discard(f1)
        if f2 == 0:
            h = f1 // 2
            deps[f1] = (h, f1 - h)
            for o in (h, f1 - h):
                if o and o not in have and o not in deps:
                    pend.add(o)
            continue
        q, r = divmod(f1, f2)
        if q == 1:
            deps[f1] = (f2, r)
            if r not in have and r not in deps:
                pend.add(r)
        elif q <= QCAP:
            seq = coef_chain(q)
            for i in range(1, len(seq)):
                cc = seq[i]
                if cc == seq[i - 1] * 2:
                    a, b = seq[i - 1], seq[i - 1]
                else:
                    a, b = seq[i - 1], 1
                deps[cc * f2] = (a * f2, b * f2)
            if r:
                deps[f1] = (q * f2, r)
                if r not in have and r not in deps:
                    pend.add(r)
            # r == 0: f1 == q*f2 was just recorded above
        else:
            h = f1 // 2
            deps[f1] = (h, f1 - h)
            for o in (h, f1 - h):
                if o and o not in have and o not in deps:
                    pend.add(o)
    return deps


def emit_from_deps(deps):
    idx = {1: 0}
    ops = []
    for v in sorted(deps):
        a, b = deps[v]
        ops.append((idx[a], idx[b]))
        idx[v] = len(idx)
    return ops


def recdiff_ops(targets):
    have = {1: 0}
    ops = []

    def ensure(v):
        if v in have:
            return
        g = max(h for h in have if h < v)
        d = v - g
        if d <= g:
            ensure(d)
            ops.append((have[g], have[d]))
        else:
            h = v // 2
            ensure(h)
            ensure(v - h)
            ops.append((have[h], have[v - h]))
        have[v] = len(have)

    for t in sorted(targets):
        ensure(t)
    return ops


def main():
    data = sys.stdin.read().split()
    K = int(data[0])
    targets = [int(x) for x in data[1:1 + K]]

    ops_a = emit_from_deps(boscoster_deps(targets))
    ops_b = recdiff_ops(targets)
    ops = ops_a if len(ops_a) <= len(ops_b) else ops_b

    out = [str(len(ops))]
    out += ["%d %d" % (a, b) for (a, b) in ops]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
