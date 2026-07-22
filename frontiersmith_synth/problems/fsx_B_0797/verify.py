import sys
from collections import defaultdict

VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz")


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def compress_tokens(doc, by_len):
    """Deterministic greedy LONGEST-PREFIX-MATCH tokenizer: at every position,
    use the longest dictionary entry that matches as a prefix; otherwise emit
    one literal-character token. Returns the number of tokens (= compressed
    size). Entries are distinct strings, so a length match is never ambiguous
    -- only the scan order over candidate lengths (longest first) matters, and
    that order is fixed."""
    n = len(doc)
    i = 0
    tokens = 0
    lens_desc = sorted(by_len.keys(), reverse=True)
    while i < n:
        matched = False
        for L in lens_desc:
            if L > n - i:
                continue
            if doc[i:i + L] in by_len[L]:
                i += L
                tokens += 1
                matched = True
                break
        if not matched:
            i += 1
            tokens += 1
    return tokens


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        D = int(next(it))
        K = int(next(it))
        M = int(next(it))
        minlen = int(next(it))
        maxlen = int(next(it))
        docs = [next(it) for _ in range(D)]
    except Exception:
        fail("bad input")
        return

    # ---- internal baseline B: the empty dictionary (no compression at all).
    # Every character becomes a literal token, so every document's ratio is
    # exactly 1.0 -- a trivial, always-feasible, positive construction. ----
    B = 1.0

    # ---- parse participant output ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
        return
    if not raw:
        fail("empty output")
        return
    try:
        E = int(raw[0])
    except Exception:
        fail("bad entry count")
        return
    if E < 0:
        fail("negative entry count")
        return
    entries = raw[1:1 + E]
    if len(entries) != E:
        fail("declared %d entries but only %d present" % (E, len(entries)))
        return

    total_len = 0
    seen = set()
    by_len = defaultdict(set)
    for s in entries:
        if not (minlen <= len(s) <= maxlen):
            fail("entry %r length outside [%d,%d]" % (s, minlen, maxlen))
            return
        if any(ch not in VALID_CHARS for ch in s):
            fail("entry %r has invalid characters" % (s,))
            return
        if s in seen:
            fail("duplicate entry %r" % (s,))
            return
        seen.add(s)
        by_len[len(s)].add(s)
        total_len += len(s)

    if E > M:
        fail("entry count %d exceeds M=%d" % (E, M))
        return
    if total_len > K:
        fail("dictionary budget %d exceeds K=%d" % (total_len, K))
        return

    # ---- compress every document, objective = MINIMUM ratio across all D ----
    worst = None
    for doc in docs:
        tok = compress_tokens(doc, by_len)
        ratio_doc = len(doc) / tok
        if worst is None or ratio_doc < worst:
            worst = ratio_doc

    F = worst
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
