#!/usr/bin/env python3
# Deterministic checker for dfa-transition-tour-cover (format C, MINIMIZE total
# test-suite length). CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys
from collections import deque

MAX_LINES = 200000
MAX_SYMBOLS = 2000000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    # ---- instance ------------------------------------------------------
    try:
        itxt = open(sys.argv[1]).read().split("\n")
    except Exception:
        fail("bad instance")
    try:
        head = itxt[0].split()
        n, k, s0 = int(head[0]), int(head[1]), int(head[2])
        symbols = itxt[1].split()
        if len(symbols) != k:
            fail("bad instance alphabet")
        trans = []
        for i in range(n):
            row = [int(x) for x in itxt[2 + i].split()]
            if len(row) != k:
                fail("bad instance row")
            trans.append(row)
    except Exception:
        fail("bad instance parse")
    if not (0 <= s0 < n):
        fail("bad instance start state")
    for i in range(n):
        for v in trans[i]:
            if not (0 <= v < n):
                fail("bad instance target")
    sym_index = {c: j for j, c in enumerate(symbols)}

    # ---- participant output ---------------------------------------------
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("no output")
    lines = raw.splitlines()
    if not lines:
        fail("empty output")
    try:
        m = int(lines[0].strip())
    except Exception:
        fail("bad m")
    if m < 0 or m > MAX_LINES:
        fail("m out of range")
    if len(lines) != 1 + m:
        fail("line count mismatch (expected exactly m string lines after m)")

    strings = lines[1:1 + m]
    total_len = sum(len(s) for s in strings)
    if total_len > MAX_SYMBOLS:
        fail("output too large")
    for s in strings:
        for ch in s:
            if ch not in sym_index:
                fail("character '%s' not in alphabet" % ch)

    # ---- simulate: mark every (state, symbol) transition exercised -------
    covered = [[False] * k for _ in range(n)]
    for s in strings:
        cur = s0
        for ch in s:
            j = sym_index[ch]
            covered[cur][j] = True
            cur = trans[cur][j]

    missing = 0
    for i in range(n):
        for j in range(k):
            if not covered[i][j]:
                missing += 1
    if missing > 0:
        fail("%d/%d transitions never exercised" % (missing, n * k))

    F = total_len
    if F <= 0:
        fail("zero-length suite cannot cover a non-empty transition set")

    # ---- internal baseline: cover every transition with its OWN string,
    # each the shortest path (in symbols) from s0 to that transition's source
    # state followed by the transition's symbol. Highly redundant (restarts
    # from s0 every time) but always feasible. ---------------------------
    dist = [-1] * n
    dist[s0] = 0
    dq = deque([s0])
    while dq:
        u = dq.popleft()
        for j in range(k):
            v = trans[u][j]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)
    B = 0
    for i in range(n):
        for j in range(k):
            # dist[i] is finite: the backbone construction guarantees every
            # state is reachable from s0, but guard defensively anyway.
            d = dist[i] if dist[i] >= 0 else n
            B += d + 1

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("n=%d k=%d F=%d B=%d Ratio: %.6f" % (n, k, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
