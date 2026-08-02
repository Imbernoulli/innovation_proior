#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Claim Ring
Audit problem (fsx_B_1267, format C). Prints "... Ratio: <float in [0,1]>".

The participant submits a budget-feasible SET of claim indices to
investigate. We re-derive the hidden fraud/ring ground truth from the same
seeded construction gen.py used (ring_truth.build), never shown on stdin,
and score the TRUE fraudulent value recovered by the submitted set. A
selection that only exploits the visible per-claim plausibility score will
catch lone sloppy fraud but miss the planted collusion rings, whose
member claims are drawn from the identical plausibility band as ordinary
business.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_truth


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        raw_in = open(in_path).read()
    except Exception:
        fail("cannot read input")

    try:
        itoks = raw_in.split()
        it = iter(itoks)
        N = int(next(it)); M = int(next(it))
        NC = int(next(it)); NP = int(next(it)); NA = int(next(it))
        test_id = int(next(it))
        for _ in range(N):
            next(it); next(it); next(it); next(it); next(it); next(it)
    except Exception:
        fail("bad instance (should never happen)")

    if not (1 <= test_id <= 10):
        fail("bad testId in instance")

    # Re-derive full ground truth (public fields + hidden fraud/ring labels)
    # from the SAME deterministic construction gen.py used, and cross-check
    # it reproduces this exact input before trusting anything.
    truth = ring_truth.build(test_id)
    if truth["N"] != N or truth["budget"] != M or truth["NC"] != NC \
            or truth["NP"] != NP or truth["NA"] != NA:
        fail("instance/testId mismatch (should never happen)")
    chk = ["%d %d %d %d %d %d" % (truth["N"], truth["budget"], truth["NC"],
                                   truth["NP"], truth["NA"], test_id)]
    for c in truth["claims"]:
        chk.append("%d %d %d %.2f %.4f %d" % (
            c["claimant"], c["provider"], c["adjuster"], c["amount"],
            c["plausibility"], c["cost"]))
    if "\n".join(chk) + "\n" != raw_in:
        fail("instance does not match the deterministic generator (tampered .in?)")

    claims = truth["claims"]

    # ---- parse participant output: "K idx_1 idx_2 ... idx_K", tokens only
    # (whitespace/newlines interchangeable); nothing may trail after. ----
    try:
        otoks = open(out_path).read().split()
    except Exception:
        fail("no output")
    if not otoks:
        fail("empty output")
    try:
        K = int(otoks[0])
    except Exception:
        fail("bad count token")
    if K < 0 or K > N:
        fail("investigated count %d out of range [0,%d]" % (K, N))
    if len(otoks) < 1 + K:
        fail("truncated output: need %d index tokens, got %d" % (K, len(otoks) - 1))
    if len(otoks) > 1 + K:
        fail("trailing garbage after expected %d tokens" % (1 + K))

    idxs = []
    seen = set()
    for t in otoks[1:1 + K]:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric claim index %r" % t)
        if not math.isfinite(v):
            fail("non-finite claim index %r" % t)
        iv = int(v)
        if iv != v:
            fail("non-integer claim index %r" % t)
        if iv < 0 or iv >= N:
            fail("claim index %d out of range [0,%d)" % (iv, N))
        if iv in seen:
            fail("duplicate claim index %d" % iv)
        seen.add(iv)
        idxs.append(iv)

    total_cost = sum(claims[i]["cost"] for i in idxs)
    if total_cost > M:
        fail("investigation cost %d exceeds budget %d" % (total_cost, M))

    F = sum(claims[i]["amount"] for i in idxs if claims[i]["fraud"])

    # ---- checker's own internal baseline: audit claims in the order the
    # instance lists them (a data-blind, order-only pass -- it uses no
    # amount, plausibility, or party information whatsoever), filling the
    # budget as it goes. Since claim order carries no information (it was
    # shuffled at construction time), this recovers only the "ambient"
    # share of hidden fraud value proportional to the budget's coverage. ----
    b_cost = 0
    B = 0.0
    for i in range(N):
        c = claims[i]["cost"]
        if b_cost + c > M:
            continue
        b_cost += c
        if claims[i]["fraud"]:
            B += claims[i]["amount"]
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print("recovered=%.4f baseline=%.4f budget_used=%d/%d Ratio: %.6f"
          % (F, B, total_cost, M, sc / 1000.0))


if __name__ == "__main__":
    main()
