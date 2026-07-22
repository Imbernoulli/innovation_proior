import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def ambiguity_sum(fingerprints):
    """Given a list of integer fingerprints (one per hypothesis in H, same order for all
    callers), return sum over hypotheses of |{h' : fp[h']==fp[h]}| = sum over fingerprint
    buckets of size^2. This is the candidate-fault-set ambiguity total (lower is better)."""
    buckets = {}
    for fp in fingerprints:
        buckets[fp] = buckets.get(fp, 0) + 1
    total = 0
    for fp in fingerprints:
        total += buckets[fp]
    return total


def main():
    inp = open(sys.argv[1]).read().split()
    out_text = open(sys.argv[2]).read()

    try:
        it = iter(inp)
        n = int(next(it))
        m = int(next(it))
        cap = int(next(it))
        K = int(next(it))
        pairs = []
        for _ in range(K):
            a = int(next(it)); b = int(next(it))
            pairs.append((a, b))
    except Exception:
        fail("bad input")

    # ---- internal baseline B: block-partition probe matrix (naive, no bit-encoding) ----
    # Split the n components into m contiguous blocks; probe i tests block i as a whole.
    # A single component's fingerprint is then just "which block it is in" (one-hot),
    # so within a block every component (and every pair inside it) is fully aliased.
    nb = max(1, min(m, n))
    block_size = (n + nb - 1) // nb
    blk_of = [min(j // block_size, nb - 1) for j in range(n)]
    base_fp_single = [1 << blk_of[j] for j in range(n)]
    B_finger = list(base_fp_single)
    for (a, b) in pairs:
        B_finger.append(base_fp_single[a] | base_fp_single[b])
    B = ambiguity_sum(B_finger)
    B = max(1, B)

    # ---- parse participant output: r, then r lines of an n-character '0'/'1' string ----
    # Strict format: split on '\n', drop ONLY the single trailing empty entry produced
    # by a final newline (not blank lines anywhere else), and never strip row content --
    # an extra blank row or whitespace-padded row is a real format violation.
    lines = out_text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        fail("empty output")
    try:
        r = int(lines[0].strip())
    except Exception:
        fail("bad r")
    if r < 0 or r > m:
        fail("r out of [0,%d]" % m)
    if len(lines) != 1 + r:
        fail("expected exactly %d line(s) (r=%d header + %d rows), got %d" % (1 + r, r, r, len(lines)))

    rows = lines[1:1 + r]
    fp = [0] * n  # fp[j] = bitmask over probes containing component j
    for pi, row in enumerate(rows):
        if len(row) != n:
            fail("row %d has length %d != n=%d" % (pi, len(row), n))
        weight = 0
        bit = 1 << pi
        for j, ch in enumerate(row):
            if ch == '1':
                weight += 1
                if weight > cap:
                    fail("row %d exceeds probe cap %d" % (pi, cap))
                fp[j] |= bit
            elif ch != '0':
                fail("row %d has non-binary char %r" % (pi, ch))

    # non-finite/garbage tokens (nan, inf, huge numbers, letters) cannot survive the
    # strict per-character '0'/'1' row check above, so no separate finiteness pass is
    # needed on the participant side.

    # ---- objective: sum of candidate-fault-set sizes over the sweep = n singles + K pairs ----
    finger = list(fp)  # singles
    for (a, b) in pairs:
        finger.append(fp[a] | fp[b])

    F = ambiguity_sum(finger)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("n=%d K=%d r=%d F=%d B=%d Ratio: %.6f" % (n, K, r, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
