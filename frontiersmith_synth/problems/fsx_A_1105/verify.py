import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    # ---------- read instance ----------
    try:
        inp = open(sys.argv[1], "rb").read().split()
    except Exception:
        fail("cannot read input")
    try:
        n = int(inp[0]); K = int(inp[1]); c = int(inp[2]); L = int(inp[3]); q = int(inp[4])
        S = inp[5].decode("ascii")
    except Exception:
        fail("bad input header")
    if len(S) != n or n < 1:
        fail("bad input string length")
    base = [ord(ch) - 97 for ch in S]
    if any(t < 0 or t >= q for t in base):
        fail("bad input alphabet")

    # ---------- read participant output (bounded) ----------
    try:
        raw = open(sys.argv[2], "rb").read(1 << 20).split()
    except Exception:
        fail("cannot read output")
    if not raw:
        fail("empty output")
    try:
        m = int(raw[0])
    except Exception:
        fail("bad rule count token")
    if m < 0 or m > K:
        fail("rule count out of range [0,%d]" % K)
    toks = raw[1:]
    if len(toks) != m:
        fail("rule count mismatch: declared %d, found %d" % (m, len(toks)))

    maxrhs_raw = 4 * L + 8
    axiom = list(base)
    rules = []
    for i in range(m):
        try:
            s = toks[i].decode("ascii")
        except Exception:
            fail("bad rule encoding")
        if len(s) < 2 or len(s) > maxrhs_raw:
            fail("rule %d raw length out of range" % (i + 1))
        pat = []
        j = 0
        ok = True
        while j < len(s):
            ch = s[j]
            if ch == '#':
                j += 1
                d0 = j
                while j < len(s) and s[j].isdigit():
                    j += 1
                if j == d0:
                    ok = False
                    break
                ref = int(s[d0:j])
                if ref < 1 or ref > i:   # only earlier rules may be referenced
                    ok = False
                    break
                pat.append(q + ref - 1)
            else:
                o = ord(ch) - 97
                if o < 0 or o >= q:
                    ok = False
                    break
                pat.append(o)
                j += 1
        if not ok or len(pat) == 0:
            fail("rule %d has bad tokens" % (i + 1))
        if len(pat) < 2 or len(pat) > L:
            fail("rule %d token length out of range [2,%d]" % (i + 1, L))

        # apply: left-to-right, non-overlapping, all occurrences, irreversible
        lp = len(pat)
        new = []
        pos = 0
        occ = 0
        la = len(axiom)
        while pos <= la - lp:
            if axiom[pos:pos + lp] == pat:
                new.append(q + i)
                pos += lp
                occ += 1
            else:
                new.append(axiom[pos])
                pos += 1
        new.extend(axiom[pos:])
        if occ == 0:
            fail("rule %d matches nothing" % (i + 1))
        axiom = new
        rules.append(pat)

    F = len(axiom) + sum(len(p) + c for p in rules)
    if F <= 0:
        fail("nonpositive grammar size")
    B = n  # internal baseline: zero rules, grammar = the raw axiom
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("grammar_size=%d baseline=%d rules=%d Ratio: %.6f" % (F, B, m, sc / 1000.0))


main()
