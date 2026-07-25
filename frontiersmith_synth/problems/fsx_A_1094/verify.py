#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for fsx_A_1094.

Replays the submitted move sequence move-by-move, validating strict legality
at every step, then scores the collected weight of the FINAL arrangement.
Baseline B = collected weight of the initial arrangement (the do-nothing
construction), so doing nothing scores exactly 0.1 and 10x better caps at 1.0.
"""
import sys
from collections import Counter

MAXMOVES = 400

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def coverage(cnt, w):
    return sum(w[c - 1] for c, k in cnt.items() if k > 0)

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("io")

    # ---- parse instance ----
    try:
        it = iter(inp)
        N = int(next(it)); M = int(next(it))
        w = [int(next(it)) for _ in range(N)]
        pos = [int(next(it)) for _ in range(M)]
    except Exception:
        fail("bad input")
    if not (1 <= N <= 60 and 1 <= M <= 30):
        fail("bad input bounds")
    if any(x < 0 or x > 10**6 for x in w):
        fail("bad weights")
    if any(p < 1 or p > N for p in pos):
        fail("bad positions")

    cnt = Counter(pos)
    B = coverage(cnt, w)          # internal baseline: do nothing
    if B <= 0:
        fail("degenerate instance")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        k = int(out[0])
    except Exception:
        fail("bad move count")
    if k < 0 or k > MAXMOVES:
        fail("move count out of range")
    if len(out) != 1 + 2 * k:
        fail("token count mismatch")

    moves = []
    for j in range(k):
        t = out[1 + 2 * j]
        v = out[2 + 2 * j]
        if t not in ("s", "g"):
            fail("bad move token %r" % t)
        try:
            i = int(v)
        except Exception:
            fail("bad move index %r" % v)
        if not (2 <= i <= N - 1):
            fail("move index %d out of range" % i)
        moves.append((t, i))

    # ---- replay ----
    for t, i in moves:
        if t == "s":                                   # scatter at i
            if cnt.get(i, 0) < 2:
                fail("illegal scatter at %d" % i)
            cnt[i] -= 2
            if cnt[i] == 0:
                del cnt[i]
            cnt[i - 1] = cnt.get(i - 1, 0) + 1
            cnt[i + 1] = cnt.get(i + 1, 0) + 1
        else:                                          # gather at i
            if cnt.get(i - 1, 0) < 1 or cnt.get(i + 1, 0) < 1:
                fail("illegal gather at %d" % i)
            cnt[i - 1] -= 1
            if cnt[i - 1] == 0:
                del cnt[i - 1]
            cnt[i + 1] -= 1
            if cnt[i + 1] == 0:
                del cnt[i + 1]
            cnt[i] = cnt.get(i, 0) + 2

    F = coverage(cnt, w)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
