#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  (ans ignored) -- deterministic grader for the
pollution-attribution-wind problem.

Reads only the test id from <in>'s header, then regenerates the hidden world
(candidate factory coordinates + SECRET true emission rates + background law)
and a FAIR, direction-diverse HELD-OUT day set entirely from that id (identical
code path to gen.py's, via common.py -- never printed on either stdout).

Parses the participant's K emission-rate estimates from <out>. Feasibility:
exactly K tokens, each a finite number in [0, UPPER_BOUND]; anything else ->
Ratio: 0.0.

Objective (minimized internally, then inverted into the printed Ratio) is a
combined relative-L1 ERROR: 0.7 * (error recovering the true rate vector,
itself split 0.6/0.4 between the colocated-cluster sources and the standalone
sources -- resolving the clusters is the point of the instance) + 0.3 *
(error predicting the held-out days' concentrations under the SAME transport
law). Baseline B is that SAME error evaluated at the checker's own trivial
construction (assume every source emits the population-mean rate).
Ratio = min(1000, 100*B/F) / 1000 -- smaller submitted error than the trivial
guess pushes the ratio up without bound (capped at 1.0); matching the trivial
guess's error scores ~0.1.
"""
import sys
import common

UPPER_BOUND = common.UPPER_BOUND
CLIP = 6.0    # cap on relative error terms so one wild source can't blow up the score
EPS = 2.0e-3  # floor on the error terms so a fluky near-zero error can't blow up the ratio


def fail(reason):
    print("infeasible: %s -- Ratio: 0.0" % reason)
    sys.exit(0)


def parse_test_id(inf):
    try:
        with open(inf) as fh:
            header = fh.readline().split()
        return int(header[0])
    except Exception:
        fail("bad instance header")


def parse_submission(outf, K):
    try:
        with open(outf, "r") as fh:
            raw = fh.read(2_000_000)
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) != K:
        fail("expected %d numbers, got %d" % (K, len(toks)))
    vals = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            fail("non-numeric token %r" % t)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value")
        if v < 0.0:
            fail("negative emission rate")
        if v > UPPER_BOUND:
            fail("emission rate exceeds bound")
        vals.append(v)
    return vals


def rel_l1_error(a, b):
    num = sum(abs(x - y) for x, y in zip(a, b))
    den = sum(abs(y) for y in b)
    return min(CLIP, num / max(1e-9, den))


def recovery_error(E_sub, world):
    E_true = world["E_true"]
    cluster_idx = sorted(i for grp in world["cluster_member_idx"] for i in grp)
    if not cluster_idx:
        return rel_l1_error(E_sub, E_true)
    standalone_idx = [i for i in range(world["K"]) if i not in set(cluster_idx)]
    err_cluster = rel_l1_error([E_sub[i] for i in cluster_idx], [E_true[i] for i in cluster_idx])
    if not standalone_idx:
        return err_cluster
    err_standalone = rel_l1_error([E_sub[i] for i in standalone_idx],
                                   [E_true[i] for i in standalone_idx])
    return 0.75 * err_cluster + 0.25 * err_standalone


def combined_error(E_sub, world, holdout):
    err_recov = recovery_error(E_sub, world)

    preds, truths = [], []
    for (day_id, wd, ws, y_true) in holdout:
        total = 0.0
        for (sx, sy), e in zip(world["sources"], E_sub):
            total += common.kernel(sx, sy, wd, ws) * e
        y_pred = common.dilution(ws) * total + common.background(
            day_id, world["A0"], world["A1"], world["P"])
        preds.append(y_pred)
        truths.append(y_true)
    err_hold = rel_l1_error(preds, truths)
    return max(EPS, 0.85 * err_recov + 0.15 * err_hold)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    test_id = parse_test_id(inf)
    if test_id not in common.CONFIGS:
        fail("bad test id")

    world = common.build_world(test_id)
    holdout = common.holdout_days(test_id, world)
    K = world["K"]

    E_sub = parse_submission(outf, K)

    F = combined_error(E_sub, world, holdout)
    mean_e = sum(world["E_true"]) / K
    E_base = [mean_e] * K
    B = combined_error(E_base, world, holdout)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
