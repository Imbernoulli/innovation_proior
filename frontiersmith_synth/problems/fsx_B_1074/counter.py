import sys

# Format D checker -- guild clearinghouse settlement.
#   1) Parse n obligations (debtor, creditor, amount) from <in>, net them into a
#      per-party balance[i] (positive = net creditor / owed money; negative = net debtor).
#   2) Parse the participant's settlement transfer list from <out>:  k, then k lines "u v a"
#      meaning party u pays party v amount a.
#   3) EXACT feasibility gate: applying every transfer must bring every party's
#      (received - paid) exactly equal to balance[i]. Any violation -> Ratio: 0.0.
#   4) Objective (minimize) = k, the transfer count (the "op count").
#      Baseline B = checker's own trivial construction: a single ascending-party-id
#      prefix-carry chain (feasible, but blind to any zero-sum grouping).
#      Ratio = min(1, 0.1 * B / k).

MAXK = 20000  # generous cap on submitted transfer-line count (sanity/DoS guard)
MAXAMT = 10 ** 15
MAXIOU = 10 ** 6  # per-statement bound on a single input IOU amount


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    try:
        out_text = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output file")
        return
    out = out_text.split()

    it = iter(inp)
    try:
        n = int(next(it))
        m = int(next(it))
    except Exception:
        fail("bad header")
        return
    if n < 2 or m < 0:
        fail("bad n/m")
        return

    balance = [0] * (n + 1)
    try:
        for _ in range(m):
            d = int(next(it))
            c = int(next(it))
            a = int(next(it))
            if not (1 <= d <= n and 1 <= c <= n and d != c and 0 < a <= MAXIOU):
                fail("malformed obligation record")
                return
            balance[d] -= a
            balance[c] += a
    except Exception:
        fail("bad obligation stream")
        return

    # NOTE: an all-zero balance vector is a legitimate (if degenerate) instance --
    # the statement allows k=0 in exactly that case -- so it is NOT rejected here.

    # ---- parse participant output ----
    if not out:
        fail("empty output")
        return
    try:
        k = int(out[0])
    except Exception:
        fail("bad transfer count")
        return
    if k < 0:
        fail("k < 0")
        return
    if k > MAXK:
        fail("k too large")
        return
    need = 1 + 3 * k
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))
        return

    toks = out[1:need]
    recv = [0] * (n + 1)
    paid = [0] * (n + 1)
    p = 0
    try:
        for _ in range(k):
            us = toks[p]; vs = toks[p + 1]; as_ = toks[p + 2]
            p += 3
            # reject non-finite / non-integer tokens explicitly (nan/inf/1e3 all fail int())
            u = int(us); v = int(vs); amt = int(as_)
            if not (1 <= u <= n and 1 <= v <= n):
                fail("party id out of range")
                return
            if u == v:
                fail("self-transfer")
                return
            if not (0 < amt <= MAXAMT):
                fail("transfer amount out of range")
                return
            paid[u] += amt
            recv[v] += amt
    except (ValueError, IndexError):
        fail("non-integer / malformed transfer token")
        return

    for i in range(1, n + 1):
        if recv[i] - paid[i] != balance[i]:
            fail("party %d not settled (net %d != required %d)" % (i, recv[i] - paid[i], balance[i]))
            return

    F = k  # your objective: number of settlement transfers

    # ---- checker's own trivial baseline: ascending-id prefix-carry chain ----
    B = 0
    carry = 0
    for i in range(1, n):
        carry += balance[i]
        if carry != 0:
            B += 1
    if B == 0:
        B = 1

    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("k=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
