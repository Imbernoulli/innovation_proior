#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an unused placeholder, per format-C contract)

Deterministic scorer for "consensus-log-reconcile" (fsx_C_1222).

Instance (stdin, produced by gen.py):
  R K N BUDGET
  mtype_1 .. mtype_K        (0=NONE, 1=SUM, 2=MAX -- legal merge op for each key)
  mcost_1 .. mcost_K        (positive int: budget spent to merge that key)
  N op lines:  replica key value weight timestamp vc_0 .. vc_{R-1}
     (1-indexed op id = line's position among these N lines)

A vector clock vc_a "happened-before" vc_b  iff  vc_a <= vc_b componentwise
and vc_a != vc_b. Two ops on the SAME key are "concurrent" iff neither
happened-before the other. For key k, the FRONTIER F(k) is the set of ops
touching k that are not happened-before by any other op touching k -- a
structural invariant of the (order-independent) causal history alone, and
therefore something every replica computes identically regardless of
delivery/gossip order. That is the convergence guarantee this checker tests:
any resolution defined purely in terms of F(k) converges; anything else does
not correctly reconstruct "what every replica must agree on".

Artifact (stdout): exactly K lines (order irrelevant), each 4 tokens:
  key_id mode ref value
    mode 'P': ref must be an op id that touches key_id; value must equal
              that op's value exactly. Credit = op's weight IF the op is in
              F(key_id), else 0 (a valid but causally-superseded pick).
    mode 'M': legal only if mtype[key_id] != NONE and |F(key_id)| >= 2.
              value must equal the exact merge of {op.value : op in F(key_id)}
              (sum for SUM, max for MAX). Credit = sum of weights of F(key_id).
              Costs mcost[key_id] against the global BUDGET.
Every key 0..K-1 must appear in the output exactly once. Total merge spend
must not exceed BUDGET. Any violation -> Ratio 0.0 for the whole submission.

F = sum of per-key credit (maximize).
Baseline B = checker's own trivial construction: for each key, if the
frontier has size 1 use it (forced -- that's the only valid choice); if size
>= 2, arbitrarily use the MINIMUM-weight frontier member and never merge.
Ratio = min(1000, 100*F/B) / 1000.

Pure integer arithmetic, O(N^2) worst case (N <= a few hundred); deterministic.
"""
import sys

SCALE_LO, SCALE_HI = 0.0, 1e9


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    try:
        R = int(nxt()); K = int(nxt()); N = int(nxt()); BUDGET = int(nxt())
        if R <= 0 or K <= 0 or N <= 0 or BUDGET < 0:
            raise ValueError("bad header")
        mtypes = [int(nxt()) for _ in range(K)]
        mcosts = [int(nxt()) for _ in range(K)]
        for t in mtypes:
            if t not in (0, 1, 2):
                raise ValueError("bad mtype")
        for c in mcosts:
            if c <= 0:
                raise ValueError("bad mcost")
        ops = []
        for _ in range(N):
            replica = int(nxt()); key = int(nxt()); value = int(nxt())
            weight = int(nxt()); ts = int(nxt())
            vc = [int(nxt()) for _ in range(R)]
            if not (0 <= replica < R) or not (0 <= key < K) or weight <= 0:
                raise ValueError("bad op fields")
            ops.append(dict(replica=replica, key=key, value=value, weight=weight, ts=ts, vc=vc))
    except (StopIteration, ValueError, OverflowError) as e:
        raise ValueError(f"malformed input: {e}")
    return R, K, N, BUDGET, mtypes, mcosts, ops


def leq(a, b):
    return all(x <= y for x, y in zip(a, b))


def compute_frontiers(K, ops):
    """frontiers[k] = list of 1-indexed op ids touching key k that are not
    causally dominated by another op touching key k."""
    by_key = [[] for _ in range(K)]
    for idx, op in enumerate(ops, start=1):
        by_key[op["key"]].append(idx)
    frontiers = [None] * K
    for k in range(K):
        ids = by_key[k]
        fr = []
        for i in ids:
            vci = ops[i - 1]["vc"]
            dominated = False
            for j in ids:
                if i == j:
                    continue
                vcj = ops[j - 1]["vc"]
                if leq(vci, vcj) and vci != vcj:
                    dominated = True
                    break
            if not dominated:
                fr.append(i)
        frontiers[k] = fr
    return by_key, frontiers


def parse_int_tok(tok):
    v = int(tok)  # rejects 'nan', 'inf', '3.5', '1e3', ...
    if v < -10**9 or v > 10**9:
        raise ValueError("out of range")
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        R, K, N, BUDGET, mtypes, mcosts, ops = read_instance(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    by_key, frontiers = compute_frontiers(K, ops)

    try:
        with open(out_path, "r") as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != 4 * K:
        fail(f"expected {4 * K} tokens ({K} lines of 4), got {len(raw)}")

    seen_keys = set()
    F = 0
    spend = 0
    for g in range(K):
        tok_k, tok_mode, tok_ref, tok_val = raw[4 * g], raw[4 * g + 1], raw[4 * g + 2], raw[4 * g + 3]
        try:
            k = parse_int_tok(tok_k)
        except (ValueError, OverflowError):
            fail(f"non-integer/out-of-range key token: {tok_k!r}")
        if not (0 <= k < K):
            fail(f"key {k} out of range [0,{K})")
        if k in seen_keys:
            fail(f"key {k} repeated")
        seen_keys.add(k)

        if tok_mode not in ("P", "M"):
            fail(f"bad mode token {tok_mode!r} (must be 'P' or 'M')")

        try:
            ref = parse_int_tok(tok_ref)
            value = parse_int_tok(tok_val)
        except (ValueError, OverflowError):
            fail(f"non-integer/out-of-range ref/value at key {k}")

        frontier = frontiers[k]

        if tok_mode == "P":
            if not (1 <= ref <= N):
                fail(f"key {k}: ref {ref} not a valid op id")
            op = ops[ref - 1]
            if op["key"] != k:
                fail(f"key {k}: ref {ref} does not touch this key")
            if value != op["value"]:
                fail(f"key {k}: declared value {value} != op {ref} value {op['value']}")
            if ref in frontier:
                F += op["weight"]
            # else: valid but causally-superseded pick -> 0 credit for this key
        else:  # "M"
            if mtypes[k] == 0:
                fail(f"key {k}: merge illegal (mtype=NONE)")
            if len(frontier) < 2:
                fail(f"key {k}: merge requires >=2 concurrent frontier writers, got {len(frontier)}")
            fr_vals = [ops[i - 1]["value"] for i in frontier]
            fr_weights = [ops[i - 1]["weight"] for i in frontier]
            merged = sum(fr_vals) if mtypes[k] == 1 else max(fr_vals)
            if value != merged:
                fail(f"key {k}: declared merge value {value} != computed {merged}")
            F += sum(fr_weights)
            spend += mcosts[k]

    if spend > BUDGET:
        fail(f"merge budget exceeded: spend {spend} > BUDGET {BUDGET}")

    # ---- checker's own trivial baseline construction ----
    B = 0
    for k in range(K):
        frontier = frontiers[k]
        if len(frontier) == 1:
            B += ops[frontier[0] - 1]["weight"]
        else:
            B += min(ops[i - 1]["weight"] for i in frontier)

    if B <= 0:
        fail("degenerate instance (non-positive baseline)")
    if not (F == F) or F < 0:
        fail("non-finite or negative objective")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F} B={B} spend={spend} BUDGET={BUDGET}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
