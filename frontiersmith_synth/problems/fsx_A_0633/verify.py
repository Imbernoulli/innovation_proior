#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- score a change-ringing touch.

Prints "... Ratio: <float in [0,1]>" on the last line and exits 0.
"""
import itertools
import math
import sys


def apply_call(row, call):
    row = list(row)
    for j in call:
        row[j - 1], row[j] = row[j], row[j - 1]
    return tuple(row)


def identity(n):
    return tuple(range(1, n + 1))


def mirror_row(row, n):
    return tuple(n + 1 - row[n - 1 - i] for i in range(n))


def valid_call_set(idx, n):
    if not idx:
        return False
    for a, b in zip(idx, idx[1:]):
        if b - a < 2:
            return False
    return all(1 <= x <= n - 1 for x in idx)


def all_valid_calls(n):
    idxs = list(range(1, n))
    res = []
    for r in range(1, len(idxs) + 1):
        for comb in itertools.combinations(idxs, r):
            if valid_call_set(comb, n):
                res.append(comb)
    return res


def fail(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_ints_line(line, what):
    toks = line.split()
    if not toks:
        fail(f"{what}: empty line")
    vals = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            fail(f"{what}: non-integer token {t!r}")
        if not math.isfinite(v):
            fail(f"{what}: non-finite token")
        vals.append(v)
    return vals


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        in_lines = f.read().split("\n")
    header = in_lines[0].split()
    n, Kmax = int(header[0]), int(header[1])
    pal_bonus = float(header[2])
    B = int(in_lines[1].strip())
    musical = []  # list of (weight, row-tuple)
    for i in range(B):
        toks = in_lines[2 + i].split()
        w = float(toks[0])
        row = tuple(int(x) for x in toks[1:1 + n])
        musical.append((w, row))

    try:
        with open(outf) as f:
            out_text = f.read()
    except FileNotFoundError:
        fail("no output file")

    out_lines = [ln for ln in out_text.split("\n")]
    # strip a single trailing blank line artifact but keep structure explicit
    if out_lines and out_lines[-1] == "":
        out_lines = out_lines[:-1]
    if not out_lines:
        fail("empty output")

    Ktoks = out_lines[0].split()
    if len(Ktoks := Ktoks) != 1:
        fail("first line must be a single integer K")
    try:
        K = int(Ktoks[0])
    except ValueError:
        fail("K is not an integer")
    if not (1 <= K <= Kmax):
        fail(f"K={K} out of range [1,{Kmax}]")
    if len(out_lines) < 1 + K:
        fail("fewer than K call lines present")
    if len(out_lines) > 1 + K:
        # extra non-blank trailing content is malformed
        for extra in out_lines[1 + K:]:
            if extra.strip() != "":
                fail("trailing garbage after the K call lines")

    calls_all_set = set(all_valid_calls(n))

    row = identity(n)
    rows = [row]
    calls = []
    for t in range(K):
        vals = read_ints_line(out_lines[1 + t], f"call line {t+1}")
        if any(v < 1 or v > n - 1 for v in vals):
            fail(f"call line {t+1}: index out of [1,{n-1}]")
        idx = tuple(vals)
        if list(idx) != sorted(set(idx)):
            fail(f"call line {t+1}: indices must be strictly increasing, no duplicates")
        if idx not in calls_all_set:
            fail(f"call line {t+1}: not a valid disjoint non-adjacent index set")
        row = apply_call(row, idx)
        rows.append(row)
        calls.append(idx)

    if rows[0] != identity(n):
        fail("row_0 is not rounds (internal error)")
    if rows[-1] != identity(n):
        fail("row_K is not rounds (closure violated)")
    if len(set(rows[:-1])) != K:
        fail("rows are not pairwise distinct before closure (trueness violated)")

    # ---- objective ----
    F = float(K)
    hitset = set()
    for t in range(1, K):
        rt = rows[t]
        for w, target in musical:
            if target in hitset:
                continue
            if rt == target:
                F += w
                hitset.add(target)

    P = 0
    for t in range(1, K):
        if rows[t] == mirror_row(rows[K - t], n):
            P += 1
    F += pal_bonus * P

    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative objective (internal error)")

    # ---- baseline: a simple, unremarkable construction (no musical/pal awareness) ----
    def bfs_path(src, dst, avoid, calls_all):
        if src == dst:
            return []
        avoid = avoid - {dst}
        from collections import deque
        dist = {src: 0}
        prev = {}
        q = deque([src])
        while q:
            r = q.popleft()
            if r == dst:
                break
            for c in calls_all:
                nr = apply_call(r, c)
                if nr in avoid or nr in dist:
                    continue
                dist[nr] = dist[r] + 1
                prev[nr] = (r, c)
                q.append(nr)
        if dst not in dist:
            return None
        pc = []
        cur = dst
        while cur != src:
            p, c = prev[cur]
            pc.append(c)
            cur = p
        pc.reverse()
        return pc

    calls_all = all_valid_calls(n)
    J = 2
    brow = identity(n)
    bvisited = {brow}
    bcalls = []
    for idx_i in range(1, J + 1):
        c = (idx_i,) if idx_i <= n - 1 else (1,)
        nr = apply_call(brow, c)
        if nr in bvisited:
            continue
        brow = nr
        bvisited.add(brow)
        bcalls.append(c)
    bhome = bfs_path(brow, identity(n), bvisited - {brow}, calls_all)
    if bhome is None:
        bcalls = [(1,), (1,)]
        brow = identity(n)
    else:
        for c in bhome:
            brow = apply_call(brow, c)
            bcalls.append(c)

    Kb = len(bcalls)
    browlist = [identity(n)]
    r0 = identity(n)
    for c in bcalls:
        r0 = apply_call(r0, c)
        browlist.append(r0)
    Fb = float(Kb)
    bhitset = set()
    for t in range(1, Kb):
        rt = browlist[t]
        for w, target in musical:
            if target in bhitset:
                continue
            if rt == target:
                Fb += w
                bhitset.add(target)
    Pb = 0
    for t in range(1, Kb):
        if browlist[t] == mirror_row(browlist[Kb - t], n):
            Pb += 1
    Fb += pal_bonus * Pb
    Fb = max(1e-9, Fb)

    sc = min(1000.0, 100.0 * F / Fb)
    ratio = sc / 1000.0
    print("F=%.4f Fb=%.4f K=%d P=%d Ratio: %.6f" % (F, Fb, K, P, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
