#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1028
   Family: patrol-phase-cover (format C, minimize worst blind window).

Two guards each output a closed, periodic walk on a k-petal "flower" graph
(hub node 0 shared by k cycles). A guard's walk of period P_g is a sequence
of P_g node ids; consecutive entries (cyclically, including the wrap from
the last back to the first) must be equal (a "wait" move) or connected by
an edge. The checker sweeps every (node, start-phase) pair over one full
combined period L = lcm(P1, P2) and scores the WORST time an intruder
appearing at that node at that phase would have to wait for either guard
to arrive. Lower is better (minimize).
"""
import sys, math

MAX_TOKENS = 20000


def read_instance(path):
    toks = open(path).read().split()
    k = int(toks[0]); P = int(toks[1])
    Ls = [int(toks[2 + i]) for i in range(k)]
    return k, P, Ls


def build_graph(Ls):
    """hub = 0. Petal p (1-indexed) contributes L_p - 1 private nodes.
    Returns (N, blocks, edges) where blocks[p-1] = list of that petal's
    private node ids in cycle order, edges = set of frozenset({a,b})."""
    k = len(Ls)
    offset = 1
    blocks = []
    edges = set()
    for Lp in Ls:
        priv = list(range(offset, offset + Lp - 1))
        blocks.append(priv)
        seq = [0] + priv
        for i in range(len(seq) - 1):
            edges.add(frozenset((seq[i], seq[i + 1])))
        edges.add(frozenset((seq[-1], 0)))  # closing edge back to hub
        offset += Lp - 1
    N = offset
    return N, blocks, edges


def canonical_tour(blocks, order):
    """One full efficient loop (hub -> petal -> hub -> petal -> ...),
    exploiting that each petal IS a cycle (no need to backtrack)."""
    tour = []
    for idx in order:
        tour.append(0)
        tour.extend(blocks[idx])
    return tour


def outback_tour(blocks, order):
    """Naive baseline: treats each petal as a dead-end corridor and walks
    out to the tip and back, never using the closing edge to the hub."""
    tour = []
    for idx in order:
        tour.append(0)
        b = blocks[idx]
        tour.extend(b)
        if len(b) >= 2:
            tour.extend(list(reversed(b))[1:])
    return tour


def parse_walks(text, N, P):
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, "non-integer token (nan/inf/garbage)"
    ptr = 0
    walks = []
    for g in range(2):
        if ptr >= len(vals):
            return None, f"missing P{g+1}"
        Pg = vals[ptr]; ptr += 1
        if Pg < 1 or Pg > P:
            return None, f"P{g+1}={Pg} out of range [1,{P}]"
        if ptr + Pg > len(vals):
            return None, "truncated walk"
        w = vals[ptr:ptr + Pg]; ptr += Pg
        for node in w:
            if node < 0 or node >= N:
                return None, "node id out of range"
        walks.append(w)
    if ptr != len(vals):
        return None, "trailing garbage after expected tokens"
    return walks, "ok"


def check_adjacency(w, edges):
    Pg = len(w)
    for i in range(Pg):
        a, b = w[i], w[(i + 1) % Pg]
        if a == b:
            continue  # wait move, always valid
        if frozenset((a, b)) not in edges:
            return False
    return True


def objective(N, w1, w2):
    """Space-time grid sweep: max over (node, start-phase) of the wait
    until either guard's periodic walk reaches that node."""
    P1, P2 = len(w1), len(w2)
    L = P1 * P2 // math.gcd(P1, P2)
    visits = [[] for _ in range(N)]
    for t in range(L):
        visits[w1[t % P1]].append(t)
        visits[w2[t % P2]].append(t)
    F = 0
    for v in range(N):
        vs = sorted(set(visits[v]))
        if not vs:
            return None  # unreachable -- should not happen if coverage holds
        gaps = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        gaps.append(vs[0] + L - vs[-1])
        F = max(F, max(gaps))
    return F


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    k, P, Ls = read_instance(inf)
    N, blocks, edges = build_graph(Ls)

    text = open(outf).read()
    walks, reason = parse_walks(text, N, P)
    if walks is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0
    w1, w2 = walks

    if not check_adjacency(w1, edges) or not check_adjacency(w2, edges):
        print("infeasible: illegal move (non-adjacent, non-wait step)")
        print("Ratio: 0.0")
        return 0

    covered = set(w1) | set(w2)
    if covered != set(range(N)):
        missing = sorted(set(range(N)) - covered)
        print(f"infeasible: nodes never visited by either guard: {missing[:10]}")
        print("Ratio: 0.0")
        return 0

    F = objective(N, w1, w2)
    if F is None or not math.isfinite(F):
        print("infeasible: non-finite / unreachable objective")
        print("Ratio: 0.0")
        return 0

    order = list(range(k))
    base = outback_tour(blocks, order)
    B = objective(N, base, base)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"N={N} P1={len(w1)} P2={len(w2)} F={F} baseline={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
