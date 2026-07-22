#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for forbidden-window de Bruijn tours.

Feasibility: the submitted string, read cyclically, must (a) use only digits 0..k-1,
(b) never contain a forbidden length-L window, (c) contain every allowed length-L window
at least once. Any violation -> "Ratio: 0.0".

Score: internal baseline B = length of a deliberately naive feasible construction (visit
each allowed edge in a fixed sorted order, BFS-walking back to it from wherever we are,
no locality/planning). Objective is MINIMIZE length, so
    ratio = min(1000, 100 * B / max(1e-9, n)) / 1000.0
"""
import sys, string
from collections import deque

DIGITS = string.digits
MAX_LEN = 5_000_000


def node_list(k, nm1):
    if nm1 == 0:
        return [""]
    out = [""]
    for _ in range(nm1):
        out = [p + d for p in out for d in DIGITS[:k]]
    return sorted(out)


def full_edges(k, L):
    return sorted(p + d for p in node_list(k, L - 1) for d in DIGITS[:k])


def build_out_adj(allowed_sorted):
    adj = {}
    for w in allowed_sorted:
        u, v = w[:-1], w[1:]
        adj.setdefault(u, []).append((v, w[-1]))
    for u in adj:
        adj[u].sort()
    return adj


def bfs_path_chars(src, dst, out_adj):
    if src == dst:
        return []
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v, c in out_adj.get(u, []):
            if v not in prev:
                prev[v] = (u, c)
                q.append(v)
    if dst not in prev:
        return None
    chars = []
    cur = dst
    while cur != src:
        pu, pc = prev[cur]
        chars.append(pc)
        cur = pu
    chars.reverse()
    return chars


def naive_baseline_len(k, L, allowed_sorted):
    out_adj = build_out_adj(allowed_sorted)
    act = sorted({w[:-1] for w in allowed_sorted} | {w[1:] for w in allowed_sorted})
    start = act[0]
    pos = start
    n = 0
    for w in allowed_sorted:
        u = w[:-1]
        if pos != u:
            path = bfs_path_chars(pos, u, out_adj)
            if path is None:
                return None
            n += len(path)
            pos = u
        n += 1  # traverse w itself
        pos = w[1:]
    if pos != start:
        path = bfs_path_chars(pos, start, out_adj)
        if path is None:
            return None
        n += len(path)
    return n


def fail(msg):
    print("WA: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    k, L, m = int(toks[0]), int(toks[1]), int(toks[2])
    forbidden = set(toks[3:3 + m])

    try:
        with open(outf) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
        return
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if len(lines) == 0:
        fail("empty output")
        return
    s = lines[0].strip()
    if len(lines) > 1:
        # allow trailing blank lines only; extra non-blank content is malformed
        for extra in lines[1:]:
            if extra.strip() != "":
                fail("multiple non-blank lines")
                return
    if s == "":
        fail("empty string")
        return
    n = len(s)
    if n > MAX_LEN:
        fail("output too long")
        return
    valid_chars = set(DIGITS[:k])
    for ch in s:
        if ch not in valid_chars:
            fail("invalid character %r" % ch)
            return

    full = full_edges(k, L)
    fullset = set(full)
    allowed_sorted = sorted(fullset - forbidden)
    if not allowed_sorted:
        fail("no allowed windows in instance (malformed input)")
        return

    reps = (n + L - 1) // n + 2
    ext = (s * reps)[: n + L - 1]
    windows = [ext[i:i + L] for i in range(n)]
    wset = set()
    for w in windows:
        if w in forbidden:
            fail("output contains forbidden window %s" % w)
            return
        wset.add(w)
    missing = [w for w in allowed_sorted if w not in wset]
    if missing:
        fail("missing %d allowed window(s), e.g. %s" % (len(missing), missing[0]))
        return

    B = naive_baseline_len(k, L, allowed_sorted)
    if B is None or B <= 0:
        fail("internal baseline construction failed (should not happen)")
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, float(n)))
    ratio = sc / 1000.0
    print("OK n=%d B=%d Ratio: %.6f" % (n, B, ratio))


if __name__ == "__main__":
    main()
