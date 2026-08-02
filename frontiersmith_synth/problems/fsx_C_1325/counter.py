#!/usr/bin/env python3
# counter.py <in> <out> <ans>
# Format D checker for a retrosynthesis route.
#
# The participant's <out> is a straight-line synthesis PROGRAM: a sequence of
# BUY / REACT instance definitions terminated by a ROOT line. The checker:
#   1) parses it strictly (schema, finiteness, forward-reference-only DAG-as-tree),
#   2) verifies it is an EXACT valid disconnection tree against the reaction
#      library given in <in> (every REACT instance must match a real reaction's
#      declared output + input molecule multiset; every BUY instance must be a
#      purchasable molecule; every produced instance is consumed at most once;
#      the ROOT instance's molecule type must be the target) -- any violation
#      scores 0,
#   3) propagates the required DELIVERED AMOUNT top-down from the root (1 unit
#      of target) using exact rational arithmetic: producing `a` units of a
#      reaction's output at yield y requires a/y units of EACH input, and
#      contributes reaction-cost * (a/y) to the total; buying `a` units of a
#      purchasable leaf contributes purchase_cost * a,
#   4) sums this into the total delivered cost F (lower is better; minimization),
#   5) compares against an internal baseline B built by the checker itself: the
#      route obtained by ALWAYS taking the smallest-id reaction available for
#      whatever molecule is currently needed (a naive, insight-free reference
#      construction that is always feasible because the generator guarantees a
#      pure smallest-id chain exists down to purchasable raw materials).
import sys
from fractions import Fraction as Fr

MAX_LINES = 4000
MAX_TOKENS_PER_LINE = 8


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it)); P = int(next(it))
    purchasable = {}
    for _ in range(P):
        mol = int(next(it)); cost = int(next(it))
        purchasable[mol] = Fr(cost)
    # reactions_by_output[out_mol] = list of (rid, inputs(tuple sorted-as-multiset), yield_frac, cost)
    reactions = {}
    reactions_by_output = {}
    for _ in range(M):
        rid = int(next(it)); output = int(next(it)); k = int(next(it))
        ins = [int(next(it)) for _ in range(k)]
        ypct = int(next(it)); cost = int(next(it))
        yfrac = Fr(ypct, 100)
        reactions[rid] = (output, tuple(sorted(ins)), yfrac, Fr(cost))
        reactions_by_output.setdefault(output, []).append(rid)
    for out in reactions_by_output:
        reactions_by_output[out].sort()
    return N, M, T, purchasable, reactions, reactions_by_output


def naive_baseline_cost(T, purchasable, reactions, reactions_by_output):
    # Always take the smallest-id reaction for whatever molecule is needed.
    # Bottom-up memoized cost-per-unit-delivered (per_unit[m] = cost to deliver 1 unit of m).
    memo = {}

    def per_unit(m, stack):
        if m in memo:
            return memo[m]
        if m in stack:
            fail("naive baseline hit a cycle (generator bug)")
        if m in purchasable and m not in reactions_by_output:
            memo[m] = purchasable[m]
            return memo[m]
        opts = reactions_by_output.get(m)
        if not opts:
            if m in purchasable:
                memo[m] = purchasable[m]
                return memo[m]
            fail("naive baseline: molecule %d has no producing reaction (generator bug)" % m)
        rid = opts[0]
        _out, ins, yfrac, cost = reactions[rid]
        stack.add(m)
        batch = (Fr(1) / yfrac)
        total = cost * batch
        for inp in ins:
            total += per_unit(inp, stack) * batch
        stack.discard(m)
        memo[m] = total
        return total

    return per_unit(T, set())


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inp, outp = sys.argv[1], sys.argv[2]
    N, M, T, purchasable, reactions, reactions_by_output = read_instance(inp)

    B = naive_baseline_cost(T, purchasable, reactions, reactions_by_output)
    if B <= 0:
        fail("baseline non-positive (generator bug)")

    try:
        with open(outp) as f:
            raw_lines = f.read().splitlines()
    except Exception:
        fail("cannot read output")

    if len(raw_lines) == 0:
        fail("empty output")
    if len(raw_lines) > MAX_LINES:
        fail("too many lines")

    # instance_id -> ('BUY', molecule) or ('REACT', reaction_id, [input_instance_ids])
    defs = {}
    order = []
    root_instance = None
    n_defs = 0

    for lineno, line in enumerate(raw_lines):
        line = line.strip()
        if line == "":
            continue
        toks = line.split()
        if len(toks) > MAX_TOKENS_PER_LINE:
            fail("line %d: too many tokens" % lineno)
        kind = toks[0]
        for t in toks[1:]:
            low = t.lower()
            if low in ("nan", "inf", "+inf", "-inf", "infinity", "-infinity"):
                fail("non-finite token on line %d" % lineno)
        if kind == "BUY":
            if len(toks) != 3:
                fail("malformed BUY on line %d" % lineno)
            try:
                mol = int(toks[1]); iid = int(toks[2])
            except ValueError:
                fail("non-integer BUY on line %d" % lineno)
            if iid in defs:
                fail("duplicate instance id %d" % iid)
            defs[iid] = ("BUY", mol)
            n_defs += 1
        elif kind == "REACT":
            if len(toks) not in (4, 5):
                fail("malformed REACT on line %d" % lineno)
            try:
                nums = [int(x) for x in toks[1:]]
            except ValueError:
                fail("non-integer REACT on line %d" % lineno)
            rid = nums[0]
            iid = nums[1]
            input_iids = nums[2:]
            if iid in defs:
                fail("duplicate instance id %d" % iid)
            if rid not in reactions:
                fail("unknown reaction id %d (line %d)" % (rid, lineno))
            for src in input_iids:
                if src not in defs:
                    fail("input instance %d used before definition (line %d)" % (src, lineno))
            defs[iid] = ("REACT", rid, input_iids)
            n_defs += 1
        elif kind == "ROOT":
            if len(toks) != 2:
                fail("malformed ROOT on line %d" % lineno)
            try:
                iid = int(toks[1])
            except ValueError:
                fail("non-integer ROOT on line %d" % lineno)
            if iid not in defs:
                fail("ROOT refers to undefined instance %d" % iid)
            if root_instance is not None:
                fail("multiple ROOT lines")
            root_instance = iid
        else:
            fail("unknown line kind %r on line %d" % (kind, lineno))
        if n_defs > 2000:
            fail("too many instances")

    if root_instance is None:
        fail("no ROOT line")

    def molecule_of(iid):
        d = defs[iid]
        if d[0] == "BUY":
            return d[1]
        else:
            rid = d[1]
            return reactions[rid][0]

    if molecule_of(root_instance) != T:
        fail("ROOT instance's molecule is not the target")

    # Walk the tree from ROOT, propagate required amount, enforce single-use consumption.
    consumed = set()
    total_cost = Fr(0)
    # iterative stack to avoid recursion-depth issues; each frame = (iid, amount_needed)
    stack = [(root_instance, Fr(1))]
    visited_count = 0
    while stack:
        iid, amount = stack.pop()
        visited_count += 1
        if visited_count > 4000:
            fail("tree too large")
        d = defs[iid]
        if d[0] == "BUY":
            mol = d[1]
            if mol not in purchasable:
                fail("BUY of non-purchasable molecule %d (instance %d)" % (mol, iid))
            total_cost += purchasable[mol] * amount
        else:
            rid, input_iids = d[1], d[2]
            out_mol, in_multiset, yfrac, cost = reactions[rid]
            given_mols = tuple(sorted(molecule_of(i) for i in input_iids))
            if given_mols != in_multiset:
                fail("instance %d: reaction %d input molecule mismatch" % (iid, rid))
            batch = amount / yfrac
            total_cost += cost * batch
            for src in input_iids:
                if src in consumed:
                    fail("instance %d reused as an input more than once" % src)
                consumed.add(src)
                stack.append((src, batch))

    F = total_cost
    if F <= 0:
        fail("non-positive delivered cost")

    Ff = float(F)
    Bf = float(B)
    sc = min(1000.0, 100.0 * Bf / max(1e-9, Ff))
    ratio = sc / 1000.0
    sys.stdout.write("baseline B=%.6f delivered_cost F=%.6f\n" % (Bf, Ff))
    sys.stdout.write("Ratio: %.6f\n" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
