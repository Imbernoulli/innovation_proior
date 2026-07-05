#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer for a comparator sorting network.

Format-D (op-count) checker:
  1. FIRST verify EXACT correctness: the submitted comparator network must sort
     every one of the 2^n binary inputs (zero-one principle -> sorts all inputs).
     Any violation / malformed output -> `Ratio: 0.0`.
  2. THEN count the operation cost = number of comparators F (fewer is better).
     Internal baseline B = the bubble-sort network the checker builds itself
     = n*(n-1)/2. Minimization score:  sc = min(1000, 100 * B / F);
     print Ratio = sc/1000  -> a bubble-cost answer scores 0.1.

The zero-one check is done in PARALLEL over ALL 2^n inputs using Python big
integers: wire k holds a 2^n-bit column, bit x = (x >> k) & 1. A comparator
(i, j) sets wire i = (A & B) [min] and wire j = (A | B) [max]. The network sorts
iff every adjacent pair is ordered, i.e. (wire_k & ~wire_{k+1}) == 0 for all k.
This is exact and bit-for-bit deterministic; nothing is timed.
"""
import sys

MAX_COMPARATORS = 2000   # generous cap (bubble baseline for n=22 is 231)


def fail(reason):
    # print reason (ignored by harness) then the canonical zero score
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_n(path):
    with open(path) as f:
        toks = f.read().split()
    return int(toks[0])


def read_network(path, n):
    """Parse the participant output strictly. Returns list of (i, j) comparators.
    Rejects non-integer tokens (this catches nan/inf/garbage), odd token counts,
    out-of-range indices, self-comparisons and over-long outputs."""
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) == 0:
        return []
    if len(toks) % 2 != 0:
        fail("odd number of integer tokens")
    vals = []
    for t in toks:
        # strict integer parse: reject nan, inf, floats, hex, anything non-integer
        s = t
        if s and s[0] in "+-":
            s = s[1:]
        if not s.isdigit():
            fail("non-integer token %r in output" % t)
        vals.append(int(t))
    m = len(vals) // 2
    if m > MAX_COMPARATORS:
        fail("too many comparators (%d > %d)" % (m, MAX_COMPARATORS))
    net = []
    for c in range(m):
        i, j = vals[2 * c], vals[2 * c + 1]
        if not (0 <= i < n) or not (0 <= j < n):
            fail("comparator index out of range: (%d,%d), n=%d" % (i, j, n))
        if i == j:
            fail("comparator on a single wire: (%d,%d)" % (i, j))
        net.append((i, j))
    return net


def build_masks(n):
    """masks[k] = 2^n-bit column where bit x = (x >> k) & 1, via O(n^2) doubling."""
    total = 1 << n
    masks = []
    for k in range(n):
        blk = 1 << k
        val = ((1 << blk) - 1) << blk          # one period: 2^k zeros then 2^k ones
        width = 1 << (k + 1)
        while width < total:
            val |= val << width
            width <<= 1
        masks.append(val)
    return masks


def sorts_all(net, n):
    total = 1 << n
    full = (1 << total) - 1
    masks = build_masks(n)
    for (i, j) in net:
        A = masks[i]
        B = masks[j]
        masks[i] = A & B      # min to wire i
        masks[j] = A | B      # max to wire j
    for k in range(n - 1):
        if masks[k] & ~masks[k + 1] & full:
            return False
    return True


def main():
    if len(sys.argv) < 3:
        print("usage: counter.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(1)
    inf, outf = sys.argv[1], sys.argv[2]
    n = read_n(inf)
    net = read_network(outf, n)

    if not sorts_all(net, n):
        fail("network does not sort all 2^%d binary inputs" % n)

    F = len(net)
    if F <= 0:
        fail("empty network cannot sort n>=2")
    B = n * (n - 1) // 2                       # bubble-sort baseline the checker builds
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("wires=%d comparators=%d baseline=%d" % (n, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
