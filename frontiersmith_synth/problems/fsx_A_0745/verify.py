import sys, math

HEADER_BITS = 16
TERM_COST = 3            # ceil(log2(8))
RULE_LEN_CAP = 13000
ALPHA = set("01234567")


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def ref_cost(i):
    # i = number of rules that existed strictly before this rule was defined
    if i <= 1:
        return 1
    return max(1, math.ceil(math.log2(i)))


def len_cost(L):
    # Elias-gamma-style universal code for the rule's token count L>=1:
    # 2*floor(log2(L))+1 bits -- cheap for short rules, grows for long ones.
    b = L.bit_length() - 1
    return 2 * b + 1


def main():
    try:
        inp_lines = open(sys.argv[1]).read().split("\n")
        n = int(inp_lines[0].strip())
        S = inp_lines[1]
        if len(S) != n:
            fail("bad input (internal)")
    except Exception:
        fail("bad input (internal)")

    try:
        out_text = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")
    out_lines = out_text.split("\n")

    try:
        R = int(out_lines[0].strip())
    except Exception:
        fail("bad R")

    R_CAP = 8 * n + 2000
    if R < 1 or R > R_CAP:
        fail("R out of range")
    if len(out_lines) < 1 + R:
        fail("missing rule lines")

    rules_tokens = []
    for i in range(R):
        toks = out_lines[1 + i].split()
        if not toks:
            fail("empty rule %d" % i)
        if len(toks) > RULE_LEN_CAP:
            fail("rule %d too long" % i)
        parsed = []
        for t in toks:
            if len(t) == 1 and t in ALPHA:
                parsed.append(("T", t))
            elif len(t) >= 2 and t[0] == 'r' and t[1:].isdigit():
                j = int(t[1:])
                if j < 0 or j >= i:
                    fail("bad back-reference in rule %d: %s" % (i, t))
                parsed.append(("R", j))
            else:
                fail("bad token %r in rule %d" % (t, i))
        rules_tokens.append(parsed)

    # ---- expand bottom-up, bounded work budget (prevents blow-up attacks) ----
    MAX_WORK = 60 * n + 20000
    work = 0
    expand = [None] * R
    for i in range(R):
        parts = []
        cur_len = 0
        for kind, val in rules_tokens[i]:
            if kind == "T":
                parts.append(val)
                cur_len += 1
                work += 1
            else:
                sub = expand[val]
                parts.append(sub)
                cur_len += len(sub)
                work += len(sub)
            if cur_len > n:
                fail("rule %d expansion exceeds n" % i)
            if work > MAX_WORK:
                fail("expansion work budget exceeded")
        expand[i] = "".join(parts)

    final = expand[R - 1]
    if len(final) != n or final != S:
        fail("expansion mismatch")

    # ---- bit cost ----
    F = HEADER_BITS
    for i in range(R):
        rule_cost = len_cost(len(rules_tokens[i]))
        for kind, val in rules_tokens[i]:
            if kind == "T":
                rule_cost += TERM_COST
            else:
                rule_cost += ref_cost(i)
        F += rule_cost

    B = HEADER_BITS + len_cost(n) + 3 * n
    B = max(1, B)
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("R=%d F=%.3f B=%.3f Ratio: %.6f" % (R, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
