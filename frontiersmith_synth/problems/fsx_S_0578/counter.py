import sys

# Format D checker -- minimum-comparator 1-FAULT-TOLERANT sorting network.
#
#   1) Parse n (and S) from <in>. Recompute the reference sorting-network size
#      S_ref = |Batcher(n)| independently (single source of truth).
#   2) Parse the participant's comparator list from <out>:
#         F
#         F lines, each "i j"  with 0 <= i < j < n  (min -> wire i, max -> wire j)
#   3) EQUIVALENCE GATE (via the 0/1 principle, done bit-parallel over all 2^n
#      binary inputs at once):
#         (a) the network as given must SORT, and
#         (b) the single-comparator DELETION SWEEP must hold: deleting ANY ONE
#             comparator must still leave a network that sorts.
#      Any violation -> Ratio: 0.0 (with a reason).
#   4) Objective (minimize) = F = comparator count.
#      Let E = F - S_ref (redundancy beyond a plain sort).  Duplication has E=S_ref,
#      triplication E=2*S_ref.  Score:  ratio = min(1, 0.2 * S_ref / max(1,E)),
#      and ratio = 1.0 if E <= 0 (a fault-tolerant net no larger than the reference
#      sort would be an extraordinary result).  The true minimum is unknown, so the
#      ceiling stays open.

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def batcher(n):
    net = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a = i + j
                        b = i + j + k
                        if a < n and b < n:
                            net.append((a, b))
            k //= 2
        p *= 2
    return net


def init_wires(n):
    """wire[i] = big integer whose bit t = ((t >> i) & 1), across all t in [0,2^n)."""
    size = 1 << n
    W = [0] * n
    for i in range(n):
        block = 1 << i
        val = 0
        pos = 0
        while pos < size:
            pos += block                      # 'block' zeros
            if pos < size:
                val |= (((1 << block) - 1) << pos)   # 'block' ones
                pos += block
        W[i] = val
    return W


def targets(n):
    """sorted-output target: wire[i] bit t = 1 iff popcount(t) >= n - i."""
    size = 1 << n
    pc = [bin(t).count("1") for t in range(size)]
    T = [0] * n
    for i in range(n):
        need = n - i
        tgt = 0
        for t in range(size):
            if pc[t] >= need:
                tgt |= (1 << t)
        T[i] = tgt
    return T


def sorts(net, n, W0, T):
    W = list(W0)
    for (a, b) in net:
        x = W[a]; y = W[b]
        W[a] = x & y                          # min (AND on 0/1)
        W[b] = x | y                          # max (OR on 0/1)
    for i in range(n):
        if W[i] != T[i]:
            return False
    return True


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        n = int(inp[0])
    except Exception:
        fail("bad input header")
    if not (2 <= n <= 14):
        fail("n out of supported range")

    S_ref = len(batcher(n))
    cap = 12 * S_ref                          # bound checker work on junk submissions

    out = open(sys.argv[2]).read().split()
    if not out:
        fail("empty output")
    try:
        F = int(out[0])                       # int() rejects nan/inf/floats
    except Exception:
        fail("bad comparator count")
    if F < 1:
        fail("F < 1")
    if F > cap:
        fail("too many comparators (F=%d > cap %d)" % (F, cap))
    if len(out) != 1 + 2 * F:
        fail("wrong token count (got %d, need %d)" % (len(out), 1 + 2 * F))

    net = []
    idx = 1
    for _ in range(F):
        try:
            i = int(out[idx]); j = int(out[idx + 1])
        except Exception:
            fail("non-integer comparator endpoint")
        idx += 2
        if not (0 <= i < j < n):
            fail("comparator (%d,%d) violates 0<=i<j<n" % (i, j))
        net.append((i, j))

    W0 = init_wires(n)
    T = targets(n)

    # (a) the network must sort with no fault
    if not sorts(net, n, W0, T):
        fail("network does not sort")
    # (b) single-comparator deletion sweep
    for k in range(F):
        if not sorts(net[:k] + net[k + 1:], n, W0, T):
            fail("deleting comparator #%d breaks sorting" % k)

    E = F - S_ref
    if E <= 0:
        ratio = 1.0
    else:
        ratio = min(1.0, 0.2 * S_ref / E)
    print("F=%d S=%d E=%d Ratio: %.6f" % (F, S_ref, E, ratio))


if __name__ == "__main__":
    main()
