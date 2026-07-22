#!/usr/bin/env python3
# counter.py <in> <out> <ans>   (ans is an unused placeholder)
#
# Format D (op-count-style budget) checker for "Nested Dyadic Quantization Under a Split
# Budget". FIRST verifies the submitted progressive code is a STRUCTURALLY VALID nested
# dyadic partition built by an ordered sequence of irreversible splits within a shared
# budget (wrong -> Ratio: 0.0), THEN reconstructs every target point through its final leaf
# and scores the total weighted squared error against an internal zero-split baseline.
# Deterministic; O(input size); never times anything.
import sys, math

TOKMAX = 4_000_000     # hard cap on total tokens read (adversarial guard)


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    try:
        with open(path, "r") as f:
            txt = f.read()
    except Exception:
        fail("cannot read file")
    toks = txt.split()
    if len(toks) > TOKMAX:
        fail("output too large")
    return toks


def next_int(it, lo=None, hi=None, what="int"):
    try:
        tok = next(it)
    except StopIteration:
        fail("unexpected EOF reading %s" % what)
    try:
        v = int(tok)
    except ValueError:
        fail("bad %s token '%s'" % (what, tok[:40]))
    if lo is not None and v < lo:
        fail("%s=%d below min %d" % (what, v, lo))
    if hi is not None and v > hi:
        fail("%s=%d above max %d" % (what, v, hi))
    return v


def next_float(it, what="float"):
    try:
        tok = next(it)
    except StopIteration:
        fail("unexpected EOF reading %s" % what)
    try:
        v = float(tok)
    except ValueError:
        fail("bad %s token '%s'" % (what, tok[:40]))
    if not math.isfinite(v):
        fail("non-finite %s" % what)
    if v < -1e6 or v > 1e6:
        fail("%s out of sane range" % what)
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage: counter.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read the instance ----
    with open(in_path, "r") as f:
        in_toks = iter(f.read().split())
    C = int(next(in_toks)); S = int(next(in_toks)); D = int(next(in_toks))
    channels = []
    for c in range(C):
        P = int(next(in_toks))
        pts = []
        for _ in range(P):
            v = float(next(in_toks)); w = int(next(in_toks))
            pts.append((v, w))
        channels.append(pts)

    # ---- internal baseline B: zero splits everywhere, leaf value = weighted mean ----
    B = 0.0
    for pts in channels:
        sw = sum(w for _, w in pts)
        if sw <= 0:
            continue
        mean = sum(v * w for v, w in pts) / sw
        B += sum(w * (v - mean) ** 2 for v, w in pts)
    B = max(1e-9, B)

    # ---- parse the participant artifact ----
    toks = read_tokens(out_path)
    it = iter(toks)

    K = next_int(it, lo=0, hi=S, what="K")
    active = [set([(0, 0)]) for _ in range(C)]   # per-channel active leaf set
    for _ in range(K):
        c = next_int(it, lo=0, hi=C - 1, what="split.c")
        depth = next_int(it, lo=0, hi=D - 1, what="split.depth")  # must be splittable (<D)
        maxpos = 1 << depth
        pos = next_int(it, lo=0, hi=maxpos - 1, what="split.pos")
        node = (depth, pos)
        if node not in active[c]:
            fail("split of non-active node channel=%d depth=%d pos=%d" % (c, depth, pos))
        active[c].discard(node)
        active[c].add((depth + 1, 2 * pos))
        active[c].add((depth + 1, 2 * pos + 1))

    total_leaves = sum(len(s) for s in active)
    L = next_int(it, lo=0, hi=total_leaves, what="L")
    if L != total_leaves:
        fail("leaf-declaration count %d != active-leaf count %d" % (L, total_leaves))

    leafval = [dict() for _ in range(C)]
    remaining = [set(s) for s in active]
    for _ in range(L):
        c = next_int(it, lo=0, hi=C - 1, what="leaf.c")
        depth = next_int(it, lo=0, hi=D, what="leaf.depth")
        maxpos = 1 << depth
        pos = next_int(it, lo=0, hi=maxpos - 1, what="leaf.pos")
        node = (depth, pos)
        if node not in remaining[c]:
            fail("leaf decl for non-active/duplicate node channel=%d depth=%d pos=%d" % (c, depth, pos))
        val = next_float(it, what="leaf.value")
        leafval[c][node] = val
        remaining[c].discard(node)

    for c in range(C):
        if remaining[c]:
            fail("missing leaf declarations for channel %d: %r" % (c, sorted(remaining[c])))

    # reject trailing garbage beyond the declared schema
    try:
        extra = next(it)
        fail("trailing token '%s' after declared schema" % extra[:40])
    except StopIteration:
        pass

    # ---- reconstruct and score ----
    F = 0.0
    for c in range(C):
        lv = leafval[c]
        for v, w in channels[c]:
            depth, pos = 0, 0
            steps = 0
            while (depth, pos) not in lv:
                lo = pos / (1 << depth)
                hi = (pos + 1) / (1 << depth)
                mid = (lo + hi) / 2.0
                if v < mid:
                    depth, pos = depth + 1, 2 * pos
                else:
                    depth, pos = depth + 1, 2 * pos + 1
                steps += 1
                if steps > D + 1:
                    fail("reconstruction walk exceeded max depth (internal inconsistency)")
            val = lv[(depth, pos)]
            F += w * (v - val) ** 2

    if not math.isfinite(F):
        fail("non-finite objective")

    F = max(1e-9, F)
    sc = min(1000.0, 100.0 * B / F)
    print("B=%.6f F=%.6f Ratio: %.6f" % (B, F, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
