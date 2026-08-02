#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the handshake-DFA problem.

Reads the instance (alphabet size m, legal traces L, forbidden traces F) from <in>.
Reads the participant's DFA from <out>:
    N
    b_0 b_1 ... b_{N-1}              (accept bit per state, 0/1)
    d_0,0 d_0,1 ... d_0,m-1          (N rows; row i = transitions of state i)
    ...
    d_{N-1},0 ... d_{N-1},m-1
Start state is fixed to state 0.

Feasibility: EVERY legal trace must end in an accepting state, EVERY forbidden trace
must end in a non-accepting state. Any violation, any malformed/non-finite/out-of-range
token -> "Ratio: 0.0".

Score (feasible only): the checker builds its own baseline B -- an always-safe but
UNminimized trie-plus-explicit-sink automaton for the same instance (this is also
exactly what solutions/trivial.py is expected to submit). Fewer states than B -> higher
score:
    ratio = min(1000, 100 * B / N) / 1000.0
"""
import sys

CAP_N = 20000  # hard upper bound on an accepted state count (adversarial-output guard)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def parse_instance(in_path):
    toks = read_tokens(in_path)
    pos = 0

    def nxt():
        nonlocal pos
        v = int(toks[pos])
        pos += 1
        return v

    m = nxt()
    L = []
    n_l = nxt()
    for _ in range(n_l):
        ln = nxt()
        L.append(tuple(nxt() for _ in range(ln)))
    F = []
    n_f = nxt()
    for _ in range(n_f):
        ln = nxt()
        F.append(tuple(nxt() for _ in range(ln)))
    return m, L, F


def baseline_trie_sink(m, L):
    """Trie over L plus one explicit, self-looping, non-accepting sink for every
    transition not on a legal prefix. Always feasible for any F disjoint from L."""
    children = [dict()]   # node -> {symbol: child_node}
    accept = [False]
    for tr in L:
        cur = 0
        for c in tr:
            if c in children[cur]:
                cur = children[cur][c]
            else:
                children.append(dict())
                accept.append(False)
                nid = len(children) - 1
                children[cur][c] = nid
                cur = nid
        accept[cur] = True
    n_trie = len(children)
    sink = n_trie
    N = n_trie + 1
    delta = [[0] * m for _ in range(N)]
    for u in range(n_trie):
        for c in range(m):
            delta[u][c] = children[u].get(c, sink)
    for c in range(m):
        delta[sink][c] = sink
    acc = accept + [False]
    return N, acc, delta


def fail(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    m, L, F = parse_instance(in_path)

    raw = read_tokens(out_path)
    if not raw:
        fail("empty output")

    idx = 0

    def read_int(lo, hi, what):
        nonlocal idx
        if idx >= len(raw):
            fail(f"missing token for {what}")
        tok = raw[idx]
        idx += 1
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token for {what}: {tok!r}")
        if v != v or not (lo <= v <= hi):
            fail(f"{what}={v} out of range [{lo},{hi}]")
        return v

    N = read_int(1, CAP_N, "N")
    accept = [read_int(0, 1, f"accept[{i}]") == 1 for i in range(N)]
    delta = []
    for i in range(N):
        row = [read_int(0, N - 1, f"delta[{i}][{c}]") for c in range(m)]
        delta.append(row)
    if idx != len(raw):
        fail(f"trailing garbage after full DFA ({len(raw) - idx} extra tokens)")

    def run(trace):
        s = 0
        for c in trace:
            if not (0 <= c < m):
                fail(f"instance symbol {c} out of alphabet range (bug)")
            s = delta[s][c]
        return s

    for tr in L:
        end = run(tr)
        if not accept[end]:
            fail(f"legal trace {tr} ends in non-accepting state {end}")

    for tr in F:
        end = run(tr)
        if accept[end]:
            fail(f"forbidden trace {tr} ends in ACCEPTING state {end} -- security violation")

    B, _, _ = baseline_trie_sink(m, L)
    sc = min(1000.0, 100.0 * B / max(1e-9, float(N)))
    ratio = sc / 1000.0
    print(f"OK: N={N} baseline={B}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
