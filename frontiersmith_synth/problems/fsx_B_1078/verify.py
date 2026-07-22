#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- Foundry Line Batching checker.

Input format:
  N M k H
  p_1 ... p_N
  M lines: u v   (task u must precede task v; edges form a DAG)

Output format (participant):
  m                              (0 <= m <= k, number of production lines used)
  m lines, each: L t_1 t_2 ... t_L     (1 <= L <= H; task ids in production order)

Feasibility (ANY violation -> Ratio: 0.0):
  - m parses as an int in [0,k]
  - each line's L parses as an int in [1,H]
  - every t_i is an integer in [1,N]
  - every task id used at most once across ALL lines
  - within a line, for every pair of positions a<b, task at position b must be
    reachable from task at position a in the precedence DAG (a valid total order
    consistent with precedence -- a genuine Dilworth chain)
  - no non-finite / unparsable tokens

Objective: F = sum of profits of all scheduled tasks.
Baseline B (checker's own trivial construction): the top min(k,N) individually most
profitable tasks, each run alone on its own line (always feasible, ignores structure).
Score: maximization normalization, sc = min(1000, 100*F/B), Ratio = sc/1000.
"""
import sys


def fail(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path):
    try:
        with open(path) as f:
            txt = f.read()
    except Exception as e:
        return None
    toks = txt.split()
    return toks


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = 0

    def nxt_i():
        nonlocal ip
        v = int(itoks[ip]); ip += 1
        return v

    N = nxt_i(); M = nxt_i(); k = nxt_i(); H = nxt_i()
    profit = [0] * (N + 1)
    for i in range(1, N + 1):
        profit[i] = nxt_i()
    adj = [[] for _ in range(N + 1)]
    indeg = [0] * (N + 1)
    edges = []
    for _ in range(M):
        u = nxt_i(); v = nxt_i()
        adj[u].append(v)
        indeg[v] += 1
        edges.append((u, v))

    # Topological order (input DAG is generator-controlled and acyclic).
    from collections import deque
    dq = deque([i for i in range(1, N + 1) if indeg[i] == 0])
    topo = []
    indeg2 = indeg[:]
    while dq:
        u = dq.popleft()
        topo.append(u)
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                dq.append(v)
    topo_pos = [0] * (N + 1)
    for idx, u in enumerate(topo):
        topo_pos[u] = idx

    # Reachability bitmask per node, built in reverse topological order:
    # reach[u] = OR over direct successors w of ( bit(w) | reach[w] ).
    reach = [0] * (N + 1)
    for u in reversed(topo):
        r = 0
        for w in adj[u]:
            r |= (1 << w) | reach[w]
        reach[u] = r

    def reachable(a, b):
        """True iff b is reachable from a (a strictly precedes b)."""
        return (reach[a] >> b) & 1 == 1

    # ---- parse participant output ----
    otoks = read_ints(out_path)
    if otoks is None:
        fail("cannot read output")
    if len(otoks) == 0:
        fail("empty output")

    op = 0
    n_tok = len(otoks)

    def has_next():
        return op < n_tok

    def nxt_o():
        nonlocal op
        if op >= n_tok:
            raise IndexError("EOF")
        tok = otoks[op]
        op += 1
        try:
            return int(tok)
        except ValueError:
            raise ValueError(f"non-integer token: {tok!r}")

    try:
        m = nxt_o()
    except (ValueError, IndexError) as e:
        fail(f"cannot parse m: {e}")

    if m < 0 or m > k:
        fail(f"m={m} out of range [0,{k}]")

    used = set()
    total_profit = 0
    try:
        for _line in range(m):
            L = nxt_o()
            if L < 1 or L > H:
                fail(f"line length {L} out of range [1,{H}]")
            ids = []
            for _ in range(L):
                t = nxt_o()
                if t < 1 or t > N:
                    fail(f"task id {t} out of range [1,{N}]")
                if t in used:
                    fail(f"task {t} used more than once")
                used.add(t)
                ids.append(t)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if not reachable(ids[a], ids[b]):
                        fail(f"tasks {ids[a]},{ids[b]} not a valid precedence chain "
                             f"(not comparable in correct order)")
            total_profit += sum(profit[t] for t in ids)
    except (ValueError, IndexError) as e:
        fail(f"malformed line: {e}")

    if has_next():
        # trailing garbage is tolerated only if it's whitespace-only; token split
        # already strips whitespace so any remaining token is real garbage.
        fail("trailing garbage after declared output")

    F = float(total_profit)
    if F != F or F in (float("inf"), float("-inf")):
        fail("non-finite objective")

    # ---- checker's own trivial baseline: top min(k,N) individually profitable tasks ----
    order = sorted(range(1, N + 1), key=lambda t: (-profit[t], t))
    topk = order[:min(k, N)]
    B = float(sum(profit[t] for t in topk))
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F} B={B} m={m}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
