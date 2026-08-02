#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for fsx_B_1282.

The hidden manipulator-window ground truth M is never written to <in>; it is
re-derived by replaying the SAME seeded generator (labels.generate_instance)
using only the testId printed on line 1 of <in>. This mirrors the format-E
held-out-regeneration trick, applied here to keep participant-window labels
private from the solver while remaining bit-for-bit reproducible for scoring.

Objective: Laplace-smoothed precision-weighted recall over flagged
participant-windows, subject to a hard alert-budget K (flag more than K ->
infeasible). Baseline B is the checker's own naive construction: the top-K
participant-windows by raw cancel COUNT (side- and timing-blind) -- exactly
the "flag rapid cancellers" trap a market maker also triggers.
"""
import math
import os
import sys
from labels import generate_instance

# Absolute headroom guard: no feasible flag set can ever score above
# sqrt(F_perfect(|M|)) / FLOOR_C relative to the baseline denominator below
# FLOOR_C -- see build_baseline().
FLOOR_C = 10.0


def fail(reason):
    print("INFEASIBLE: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def score_flagset(flagged, M, K):
    """Laplace-smoothed precision-weighted recall, geometric-mean compressed
    (sqrt) to keep the dynamic range between 'random' and 'perfect' sane under
    the fixed maximization normalization sc=min(1000,100*F/B)."""
    tp = len(flagged & M)
    prec = (tp + 0.5) / (len(flagged) + 1.0)
    rec = (tp + 0.5) / (len(M) + 1.0)
    return math.sqrt(prec * rec)


def build_baseline(all_pw, cancel_count, M, K):
    """Checker's own trivial reference: top-K participant-windows by raw
    CANCEL COUNT (side- and timing-blind) -- reproduced exactly by
    solutions/trivial.py. Floored against an analytic, data-independent upper
    bound (F_perfect(|M|)/FLOOR_C) so that no feasible submission -- however
    good -- can ever be pushed to the ratio cap by an unlucky, near-zero-TP
    baseline draw (anti-saturation guard, keeps headroom above the reference
    solutions on every case)."""
    order = sorted(all_pw, key=lambda pw: (-cancel_count.get(pw, 0), pw[0], pw[1]))
    baseline_flagged = set(order[:K])
    b_raw = score_flagset(baseline_flagged, M, K)
    prec_perfect = (len(M) + 0.5) / (len(M) + 1.0)
    f_perfect = prec_perfect  # score_flagset's sqrt(prec*rec) at prec=rec=prec_perfect
    b_floor = f_perfect / FLOOR_C
    return max(b_raw, b_floor)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # --- regenerate the instance (public data + hidden labels) from testId ---
    try:
        with open(in_path, "r") as f:
            first_line = f.readline().strip()
        test_id = int(first_line)
    except Exception:
        fail("cannot read testId from input")

    inst = generate_instance(test_id)
    N, W, K = inst["N"], inst["W"], inst["K"]
    M = inst["M"]
    events = inst["events"]

    # --- bounded, strict parse of the participant's output ---
    try:
        sz = os.path.getsize(out_path)
    except OSError:
        fail("missing output")
    if sz > 5_000_000:
        fail("output too large")
    try:
        with open(out_path, "r", errors="replace") as f:
            content = f.read(5_000_001)
    except Exception:
        fail("cannot read output")

    lines = content.splitlines()
    if len(lines) < 1 or lines[0].strip() == "":
        fail("empty output")

    def parse_int(tok, lo, hi):
        tok = tok.strip()
        if not tok:
            return None
        neg = tok.startswith("-")
        body = tok[1:] if neg else tok
        if not body.isdigit() or len(body) > 12:
            return None
        v = int(tok)
        if v < lo or v > hi:
            return None
        return v

    c_val = parse_int(lines[0], 0, K)
    if c_val is None:
        fail("flag count missing, non-integer, or exceeds alert budget K=%d" % K)
    C = c_val

    body_lines = [ln for ln in lines[1:] if ln.strip() != ""]
    if len(body_lines) != C:
        fail("declared %d flagged windows but %d lines follow" % (C, len(body_lines)))

    flagged = set()
    for ln in body_lines:
        toks = ln.split()
        if len(toks) != 2:
            fail("malformed flag line: %r" % ln)
        w = parse_int(toks[0], 0, W - 1)
        pid = parse_int(toks[1], 0, N - 1)
        if w is None or pid is None:
            fail("flag index out of range or non-finite: %r" % ln)
        pair = (w, pid)
        if pair in flagged:
            fail("duplicate flagged window: %r" % ln)
        flagged.add(pair)

    if len(flagged) > K:
        fail("alert budget exceeded")

    # --- objective ---
    F = score_flagset(flagged, M, K)

    # --- checker's own trivial baseline B (see build_baseline docstring) ---
    cancel_count = {}
    for (w, pid, t, side, action, size) in events:
        if action == "C":
            key = (w, pid)
            cancel_count[key] = cancel_count.get(key, 0) + 1
    all_pw = [(w, pid) for w in range(W) for pid in range(N)]
    B = build_baseline(all_pw, cancel_count, M, K)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f TP=%d flagged=%d |M|=%d K=%d" %
          (F, B, len(flagged & M), len(flagged), len(M), K))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
