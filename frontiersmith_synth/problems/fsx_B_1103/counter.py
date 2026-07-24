#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (Format D: op-count of an embroidery stitch plan)

The participant submits an op sequence:
  line 1: M            number of ops
  next M lines: "S u v"  stitch the segment {u,v} (needle must be at u or v;
                         segment must exist and be unstitched; cost 1, plus K
                         if its color differs from the loaded color; the needle
                         ends at the other endpoint)
                "J v"    jump the needle to vertex v (cost = Manhattan distance)

The needle starts at vertex 0 with no color loaded; the first color load costs K.

We FIRST verify exact coverage (every segment stitched exactly once, all ops
valid), THEN count the total cost F. Score (minimization):
  sc = min(1000, 100 * B / F), ratio = sc/1000,
where B is the checker's own baseline: stitch the segments IN INPUT ORDER,
jumping to u whenever the needle is at neither endpoint. Any feasibility
violation -> Ratio: 0.0. Fully deterministic; integer arithmetic only.
"""
import sys

MAX_OPS = 200000


def fail(reason):
    print("VIOLATION: %s  Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    data = open(sys.argv[1]).read().split()
    pos = 0

    def ni():
        nonlocal pos
        v = int(data[pos]); pos += 1
        return v

    V = ni(); E = ni(); C = ni(); K = ni()
    xs = [0] * V; ys = [0] * V
    for i in range(V):
        xs[i] = ni(); ys[i] = ni()
    ecol = {}
    ein_order = []
    for _ in range(E):
        u = ni(); v = ni(); c = ni()
        key = (u, v) if u < v else (v, u)
        ecol[key] = c
        ein_order.append((u, v, c))

    def dist(a, b):
        return abs(xs[a] - xs[b]) + abs(ys[a] - ys[b])

    # ---- internal baseline B: input order, jump to u when not adjacent ----
    B = 0
    p = 0
    col = -1
    for (u, v, c) in ein_order:
        if p == u:
            nxt = v
        elif p == v:
            nxt = u
        else:
            B += dist(p, u)
            nxt = v
        if c != col:
            B += K
            col = c
        B += 1
        p = nxt

    # ---- participant plan ----
    raw = open(sys.argv[2]).read().split()
    if not raw:
        fail("empty output")
    try:
        M = int(raw[0])
    except ValueError:
        fail("first token (op count) is not an integer")
    if M < 0 or M > MAX_OPS:
        fail("op count out of range [0,%d]" % MAX_OPS)
    # parse ops
    ops = []
    i = 1
    for _ in range(M):
        if i >= len(raw):
            fail("fewer tokens than the declared op count")
        t = raw[i]; i += 1
        if t == "J":
            if i >= len(raw):
                fail("truncated J op")
            try:
                v = int(raw[i])
            except ValueError:
                fail("non-integer jump target")
            i += 1
            ops.append(("J", v, v))
        elif t == "S":
            if i + 1 >= len(raw):
                fail("truncated S op")
            try:
                u = int(raw[i]); v = int(raw[i + 1])
            except ValueError:
                fail("non-integer stitch endpoint")
            i += 2
            ops.append(("S", u, v))
        else:
            fail("unknown op token %r" % t)
    if i != len(raw):
        fail("extra tokens after %d ops" % M)

    # ---- simulate with strict feasibility ----
    F = 0
    p = 0
    col = -1
    stitched = set()
    for (t, u, v) in ops:
        if t == "J":
            if not (0 <= u < V):
                fail("jump target %d out of range" % u)
            F += dist(p, u)
            p = u
        else:
            if not (0 <= u < V and 0 <= v < V):
                fail("stitch endpoint out of range")
            key = (u, v) if u < v else (v, u)
            if key not in ecol:
                fail("stitch of non-existent segment %d-%d" % (u, v))
            if key in stitched:
                fail("segment %d-%d stitched twice" % key)
            if p != u and p != v:
                fail("needle not at an endpoint of segment %d-%d" % (u, v))
            c = ecol[key]
            if c != col:
                F += K
                col = c
            F += 1
            stitched.add(key)
            p = v if p == u else u
    if len(stitched) != E:
        fail("coverage incomplete: %d of %d segments stitched" % (len(stitched), E))

    if B <= 0:
        B = 1  # degenerate safeguard (generator guarantees B > 0)
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("cost=%d baseline=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
