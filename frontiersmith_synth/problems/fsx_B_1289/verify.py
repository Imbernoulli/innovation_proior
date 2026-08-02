#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for capex-stage-gate.

Reads the instance, parses the participant's staging+abandonment policy from
<out>, validates it strictly, computes its EXACT expected NPV (full
enumeration over hidden project type x every module signal, weighted by the
true joint probabilities -- no sampling, no randomness), normalizes against
the checker's own internal baseline (the "always fully commit, never
abandon" policy, guaranteed positive by the instance design), and prints
`Ratio: <float in [0,1]>`.
"""
import sys


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        M = int(next(it))
        costs = [int(next(it)) for _ in range(M)]
        acc = [float(next(it)) for _ in range(M)]
        p = float(next(it))
        VG = float(next(it))
        VB = float(next(it))
        sigma = float(next(it))
        F = float(next(it))
        r = float(next(it))
    except (StopIteration, ValueError):
        raise RuntimeError("bad instance file")
    return dict(M=M, costs=costs, acc=acc, p=p, VG=VG, VB=VB, sigma=sigma, F=F, r=r)


def leaf_value(case, cum_disc, K):
    r = case['r']
    return (-cum_disc + (r ** K) * case['VG'],
            -cum_disc + (r ** K) * case['VB'])


def eval_policy(case, boundaries, decision_tables):
    """Exact expected NPV under a given partition + decision policy.
    decision_tables[k] is a list of length 2**g_k (k = 1..K-1); entry at
    index h (bit j-1 = module j's signal, 1=Good) is 1(continue)/0(abandon)."""
    costs, acc, p, sigma, F, r = (case[k] for k in
                                   ['costs', 'acc', 'p', 'sigma', 'F', 'r'])
    K = len(boundaries)
    g = [0] + boundaries

    def rec(stage_idx, h, cum_raw, cum_disc):
        modules = range(g[stage_idx - 1] + 1, g[stage_idx] + 1)
        stage_cost = F + sum(costs[j - 1] for j in modules)
        disc = r ** (stage_idx - 1)
        new_raw = cum_raw + stage_cost
        new_disc = cum_disc + disc * stage_cost
        nbits = len(list(modules))
        valG = 0.0
        valB = 0.0
        for combo in range(1 << nbits):
            newh = h
            probG = 1.0
            probB = 1.0
            for bit_i, j in enumerate(modules):
                sig = (combo >> bit_i) & 1
                if sig:
                    newh |= (1 << (j - 1))
                a = acc[j - 1]
                if sig == 1:
                    probG *= a
                    probB *= (1 - a)
                else:
                    probG *= (1 - a)
                    probB *= a
            if stage_idx == K:
                vg, vb = leaf_value(case, new_disc, K)
            else:
                decision = decision_tables[stage_idx][newh]
                if decision == 0:
                    aval = -new_disc + disc * sigma * new_raw
                    vg, vb = aval, aval
                else:
                    vg, vb = rec(stage_idx + 1, newh, new_raw, new_disc)
            valG += probG * vg
            valB += probB * vb
        return valG, valB

    vg, vb = rec(1, 0, 0, 0)
    return p * vg + (1 - p) * vb


def full_commit_baseline(case):
    return eval_policy(case, [case['M']], {})


def parse_policy(path, M):
    with open(path) as f:
        raw = f.read().split()
    it = iter(raw)

    def next_int(what):
        try:
            tok = next(it)
        except StopIteration:
            fail("truncated output (expected %s)" % what)
        # strict integer grammar: optional sign + digits only (rejects "3.0",
        # "1e5", "nan", "inf", "+inf" etc. -- no float() parsing anywhere here)
        body = tok[1:] if tok[:1] in ("+", "-") else tok
        if not body.isdigit():
            fail("non-integer token %r (expected %s)" % (tok, what))
        return int(tok)

    K = next_int("K")
    if not (1 <= K <= M):
        fail("K=%d out of range [1,%d]" % (K, M))
    boundaries = []
    prev = 0
    for i in range(K):
        gk = next_int("g_%d" % (i + 1))
        if gk <= prev or gk > M:
            fail("boundary sequence not strictly increasing / in range: g=%r" % (boundaries + [gk]))
        boundaries.append(gk)
        prev = gk
    if boundaries[-1] != M:
        fail("last boundary g_K=%d != M=%d" % (boundaries[-1], M))

    decision_tables = {}
    g = [0] + boundaries
    for k in range(1, K):
        gk = g[k]
        n = 1 << gk
        tab = []
        for h in range(n):
            try:
                tok = next(it)
            except StopIteration:
                fail("truncated decision table at stage %d (need %d entries)" % (k, n))
            if tok not in ("0", "1"):
                fail("decision token %r at stage %d must be 0 or 1" % (tok, k))
            tab.append(int(tok))
        decision_tables[k] = tab

    # strict: no leftover tokens
    leftover = list(it)
    if leftover:
        fail("extra trailing tokens after a complete, valid policy: %r" % (leftover[:5]))

    return boundaries, decision_tables


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        case = read_instance(in_path)
    except Exception as e:
        # instance file itself is always well-formed (we generated it) --
        # a failure here is a harness/environment problem, not a WA.
        print("BAD_INSTANCE: %s" % e)
        print("Ratio: 0.0")
        sys.exit(0)

    M = case['M']
    boundaries, decision_tables = parse_policy(out_path, M)

    value = eval_policy(case, boundaries, decision_tables)
    baseline = full_commit_baseline(case)
    if baseline <= 1e-9:
        # generator guarantees a positive baseline; guard anyway
        print("Ratio: 0.0")
        sys.exit(0)

    sc = 100.0 * value / baseline
    sc = max(0.0, min(1000.0, sc))
    ratio = sc / 1000.0
    print("value=%.6f baseline=%.6f K=%d boundaries=%r" % (value, baseline, len(boundaries), boundaries))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
