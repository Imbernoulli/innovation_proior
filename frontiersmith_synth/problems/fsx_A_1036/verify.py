#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the weighted-DFA-chorus
passphrase problem. ans is an unused placeholder.

Instance format (stdin, produced by gen.py):
    m A Lmax
    repeated m times:
        n start weight        # n = #states, start state index, integer weight
        k a_1 .. a_k           # k accepting state indices
        n lines, each with A ints: transition row for state 0..n-1 over symbols 0..A-1

Participant output (stdout), read as a whitespace token stream:
    L  s_1 s_2 ... s_L         # L = length of the chosen string S (0<=L<=Lmax),
                                # each s_i in [0, A-1]; nothing else may follow.

Feasibility: L in [0,Lmax], every symbol a valid int in [0,A-1], token stream has
EXACTLY L+1 tokens (bounded read, reject trailing garbage). Any violation, or a
non-finite/non-integer token -> Ratio: 0.0.

Objective: simulate every DFA on S; F = (sum of weights of DFAs that end in an
accepting state) - EPS*L (tiny strictly-dominated tie-break favoring a shorter S
when two strings tie on total weight; EPS is far smaller than any 1-unit weight gap).

Baseline B: the checker's own trivial construction -- for each DFA i run an
independent BFS over ITS transition graph for the shortest string (length <= Lmax)
reaching one of its accepting states; B = the largest weight among DFAs that have
such a reachable string (a single-DFA-only witness). Maximization normalization:
    sc = min(1000, 100*F/max(1e-9,B));  Ratio = sc/1000
"""
import sys
from collections import deque

EPS = 1e-4


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    def nxt():
        return int(next(it))
    m = nxt(); A = nxt(); Lmax = nxt()
    dfas = []
    for _ in range(m):
        n = nxt(); start = nxt(); w = nxt()
        k = nxt()
        accept = [nxt() for _ in range(k)]
        trans = []
        for _s in range(n):
            trans.append([nxt() for _a in range(A)])
        dfas.append({"n": n, "start": start, "w": w, "accept": set(accept), "trans": trans})
    return m, A, Lmax, dfas


def bfs_shortest_accept(dfa, Lmax):
    """Shortest string length (<=Lmax) reaching an accepting state in this DFA alone,
    or None if unreachable within the budget."""
    start = dfa["start"]
    if start in dfa["accept"]:
        return 0
    dist = {start: 0}
    q = deque([start])
    A = len(dfa["trans"][0])
    while q:
        s = q.popleft()
        d = dist[s]
        if d >= Lmax:
            continue
        for a in range(A):
            ns = dfa["trans"][s][a]
            if ns not in dist:
                dist[ns] = d + 1
                if ns in dfa["accept"]:
                    return d + 1
                q.append(ns)
    return None


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]
    m, A, Lmax, dfas = read_instance(in_path)

    with open(out_path) as f:
        raw = f.read().split()

    if not raw:
        fail("empty output")

    def parse_int_strict(tok):
        # int() rejects "nan"/"inf"/floats/garbage outright -> raises ValueError
        try:
            if any(c not in "-0123456789" for c in tok):
                raise ValueError
            return int(tok)
        except ValueError:
            fail(f"non-integer token '{tok}'")

    L = parse_int_strict(raw[0])
    if L < 0 or L > Lmax:
        fail(f"L={L} out of range [0,{Lmax}]")
    if len(raw) != 1 + L:
        fail(f"expected exactly {1 + L} tokens, got {len(raw)}")

    S = []
    for tok in raw[1:1 + L]:
        v = parse_int_strict(tok)
        if v < 0 or v >= A:
            fail(f"symbol {v} out of range [0,{A - 1}]")
        S.append(v)

    # simulate every DFA on S
    F = 0.0
    for d in dfas:
        s = d["start"]
        for sym in S:
            s = d["trans"][s][sym]
        if s in d["accept"]:
            F += d["w"]
    F -= EPS * L

    # baseline: best single-DFA-only construction
    candidates = []
    for d in dfas:
        dist = bfs_shortest_accept(d, Lmax)
        if dist is not None:
            candidates.append(d["w"])
    B = max(candidates) if candidates else 1.0

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = max(0.0, sc) / 1000.0
    print(f"F={F:.6f} B={B:.6f} L={L}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
