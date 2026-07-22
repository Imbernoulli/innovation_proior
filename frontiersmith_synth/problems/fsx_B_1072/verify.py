import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def canon_rotation(t):
    """Lexicographically smallest rotation of the tuple t (the necklace's canonical form)."""
    n = len(t)
    best = t
    doubled = t + t
    for r in range(1, n):
        cand = doubled[r:r + n]
        if cand < best:
            best = cand
    return best


def distinct_necklaces(seq, k):
    """seq is a list[int] treated CYCLICALLY (length L=len(seq)). Returns the number of
    distinct rotation-equivalence classes (necklaces) among all L cyclic length-k windows."""
    L = len(seq)
    doubled = seq + seq
    classes = set()
    for i in range(L):
        window = tuple(doubled[i:i + k])
        classes.add(canon_rotation(window))
    return len(classes)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        a = int(inp[0]); k = int(inp[1]); L = int(inp[2])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: the checker's own trivial construction -----------------
    # A single constant symbol repeated L times. Always feasible, always exactly 1
    # necklace class (every cyclic window is the same all-0 word), computed the same way
    # a submission's score is computed so the normalization stays apples-to-apples.
    baseline_seq = [0] * L
    B = distinct_necklaces(baseline_seq, k)
    B = max(1, B)

    # ---- parse participant output -----------------------------------------------------
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("no output")
    toks = raw.split()
    if len(toks) != 1:
        fail("expected exactly one token (the cyclic string), got %d" % len(toks))
    s = toks[0]
    if len(s) != L:
        fail("length %d != L=%d" % (len(s), L))
    if not s.isdigit():
        fail("non-digit character in output")
    seq = [ord(c) - 48 for c in s]
    if any(v < 0 or v >= a for v in seq):
        fail("digit out of range [0,%d)" % a)

    # ---- objective: distinct necklace classes among the L cyclic k-windows ------------
    F = distinct_necklaces(seq, k)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("a=%d k=%d L=%d F=%d B=%d Ratio: %.6f" % (a, k, L, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
