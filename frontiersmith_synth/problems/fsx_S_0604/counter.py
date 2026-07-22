#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (ans ignored)

Replays a vertex-elimination sequence to accumulate the full Jacobian of a
computational DAG, counting the EXACT number of scalar multiplications, and scores
fewer-ops-is-better against the checker's own forward-mode baseline.

Feasibility (any violation -> Ratio: 0.0):
  * output tokens are integers, no nan/inf, no out-of-range ids
  * the sequence is EXACTLY the set of intermediate vertices, each once
Equivalence (format D): after replaying the sequence, the residual input->output
matrix (computed in GF(P)) must equal the reference Jacobian.  Any valid full
elimination reproduces it, so a mismatch means the sequence was not a real full
elimination -> Ratio: 0.0.

Deterministic: pure integer / modular arithmetic, no time, no randomness.
"""
import sys

P = 1_000_000_007
OP_CAP = 60_000_000   # guard against pathological orders (legit tiers are far below)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    V = int(next(it)); E = int(next(it)); M = int(next(it)); N = int(next(it))
    val = {}          # (u,v) -> GF(P) derivative
    outadj = {u: {} for u in range(V)}
    inadj = {v: set() for v in range(V)}
    for _ in range(E):
        u = int(next(it)); v = int(next(it)); w = int(next(it)) % P
        outadj[u][v] = w
        inadj[v].add(u)
    inputs = list(range(M))
    outputs = list(range(V - N, V))
    intermediates = list(range(M, V - N))
    return V, M, N, inputs, outputs, intermediates, outadj, inadj


def replay(order, V, outadj0, inadj0, cap=OP_CAP):
    """Eliminate the vertices in `order`; return (ops, jac) where jac maps
    (in,out)->GF(P) via the residual out-edges.  ops = scalar multiplications.
    Returns (None, None) if the running op-count exceeds `cap`."""
    out = {u: dict(d) for u, d in outadj0.items()}
    inn = {v: set(s) for v, s in inadj0.items()}
    ops = 0
    for v in order:
        preds = list(inn[v])
        succs = list(out[v].items())
        ops += len(preds) * len(succs)
        if ops > cap:
            return None, None
        for p in preds:
            avp = out[p][v]
            op = out[p]
            for (s, avs) in succs:
                if s in op:
                    op[s] = (op[s] + avp * avs) % P
                else:
                    op[s] = (avp * avs) % P
                    inn[s].add(p)
        # remove v
        for p in preds:
            del out[p][v]
        for (s, _) in succs:
            inn[s].discard(v)
        out[v] = {}
        inn[v] = set()
    return ops, out


def jac_of(out, inputs, outputs):
    outset = set(outputs)
    J = {}
    for i in inputs:
        for s, w in out[i].items():
            if s in outset:
                J[(i, s)] = w % P
    return J


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.000000")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    V, M, N, inputs, outputs, intermediates, outadj, inadj = read_instance(inf)

    # ---- reference Jacobian + forward-mode baseline (checker's own construction) ----
    fwd = sorted(intermediates)                       # ascending id == topological
    base_ops, ref_out = replay(fwd, V, outadj, inadj)
    ref_jac = jac_of(ref_out, inputs, outputs)
    if base_ops is None or base_ops <= 0:
        base_ops = 1

    # ---- parse participant output strictly ----
    try:
        raw = open(outf).read().split()
    except Exception:
        fail("unreadable output")
    seq = []
    for tk in raw:
        # reject nan/inf/floats/garbage: must be a plain integer
        try:
            iv = int(tk)
        except ValueError:
            fail("non-integer token: %r" % tk[:16])
        seq.append(iv)

    interset = set(intermediates)
    if len(seq) != len(intermediates):
        fail("sequence length %d != #intermediates %d" % (len(seq), len(intermediates)))
    if set(seq) != interset:
        fail("sequence is not exactly the intermediate-vertex set")
    if len(set(seq)) != len(seq):
        fail("duplicate vertex in sequence")

    # ---- replay participant order: count ops + verify equivalence ----
    ops, part_out = replay(seq, V, outadj, inadj)
    if ops is None:
        fail("op-count exceeded cap (pathological order)")
    part_jac = jac_of(part_out, inputs, outputs)
    if part_jac != ref_jac:
        fail("residual Jacobian != reference (not a valid full elimination)")
    if ops <= 0:
        ops = 1

    sc = min(1000.0, 100.0 * base_ops / max(1e-9, ops))
    print("base_ops: %d  your_ops: %d" % (base_ops, ops))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
