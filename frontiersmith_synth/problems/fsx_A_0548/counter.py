#!/usr/bin/env python3
# Deterministic checker (Format D, eval_form=flops) for "xor-straight-line-share".
#
# Instance  : a target m x n GF(2) matrix M (rows = linear forms over inputs x_0..x_{n-1}).
# Artifact  : a linear straight-line program of two-input XOR gates.
#             Nodes 0..n-1 are the inputs; gate k (0-indexed) is node n+k and equals
#             XOR of two STRICTLY-earlier nodes.  m designated output nodes must realise
#             the m rows of M (over GF(2), for every input assignment, i.e. mask equality).
# Objective : MINIMISE the XOR-gate count F = G.
# Baseline B: independent per-row XOR-folds (no reuse, no cancellation) = sum_i (wt(row_i)-1).
# Score     : minimisation -> sc = min(1000, 100 * B / F); Ratio = sc/1000.
#
# Feasibility is validated STRICTLY; any violation prints "Ratio: 0.0" and exits 0.
import sys

MAXG = 2_000_000  # hard cap on declared gate count (guards memory / time)

def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)

def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    with open(inf) as f:
        itok = f.read().split()
    ip = 0
    def inext():
        nonlocal ip
        v = int(itok[ip]); ip += 1; return v
    try:
        m = inext(); n = inext()
        rows = []
        for _ in range(m):
            r = 0
            for j in range(n):
                b = inext()
                if b: r |= (1 << j)
            rows.append(r)
    except Exception:
        fail("bad instance")

    full = (1 << n) - 1

    # internal baseline B = independent folds (strictly positive by construction)
    B = 0
    for r in rows:
        w = bin(r).count("1")
        if w >= 1:
            B += (w - 1)
    if B <= 0:
        B = 1  # safety; generator guarantees weights >= 2 so this never triggers

    # ---- read participant artifact strictly ----
    try:
        with open(outf) as f:
            data = f.read().split()
    except Exception:
        fail("no output")
    if not data:
        fail("empty output")

    # every token must be a base-10 integer (rejects nan/inf/garbage)
    def as_int(tok):
        try:
            return int(tok)
        except (ValueError, TypeError):
            fail("non-integer token")

    p = 0
    def onext():
        nonlocal p
        if p >= len(data):
            fail("truncated output")
        v = as_int(data[p]); p += 1; return v

    G = onext()
    if G < 0 or G > MAXG:
        fail("gate count out of range")
    # exact token accounting: 1 (G) + 2G (gate operands) + m (outputs)
    if len(data) != 1 + 2 * G + m:
        fail("token count mismatch")

    total_nodes = n + G
    mask = [0] * total_nodes
    for i in range(n):
        mask[i] = (1 << i)
    for k in range(G):
        a = onext(); b = onext()
        node = n + k
        if not (0 <= a < node) or not (0 <= b < node):
            fail("gate operand references a non-earlier node")
        mask[node] = mask[a] ^ mask[b]

    outputs = []
    for _ in range(m):
        o = onext()
        if not (0 <= o < total_nodes):
            fail("output index out of range")
        outputs.append(o)

    # ---- exact GF(2) equivalence: realised mask must equal each target row ----
    for i in range(m):
        got = mask[outputs[i]] & full
        if got != rows[i]:
            fail("output %d realises 0x%x != target 0x%x" % (i, got, rows[i]))

    F = G
    if F <= 0:
        # zero gates but some target row has weight >= 2 is impossible; guard anyway
        fail("zero gates cannot realise the map")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("gates=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
