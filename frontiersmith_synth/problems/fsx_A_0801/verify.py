#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -> prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic scorer for the physarum-steiner-maze problem.

Reads the maze (terminals + wall obstacles) from <in>. Parses the participant's
artifact from <out>: a feedback exponent mu and a sparse list of initial-
conductance overrides on grid edges. Runs FIXED-ROUND deterministic physarum
tube-reinforcement dynamics (current-injection Kirchhoff solve for every
terminal pair each round, accumulate flux, relax conductance toward
flux^mu, clip) starting from that initial field. The final "tube network" is
every grid edge whose converged conductance is >= THRESH; it must connect all
terminals. Objective (minimize) = number of unit-length edges in that network.

Baseline B = the checker's own reference run: mu=1.0, no overrides at all (the
"do nothing" quiescent field). Minimization normalization:
    sc = min(1000.0, 100.0 * B / max(1e-9, F));  Ratio = sc / 1000.0
Any feasibility violation (bad token count, out-of-range/non-finite mu or d0,
edge not a valid grid adjacency, disconnected final network) prints
'Ratio: 0.0' and exits 0.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physlib as pl

try:
    import numpy as np
except Exception:
    np = None


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    R = int(nxt())
    C = int(nxt())
    K = int(nxt())
    terminals = []
    for _ in range(K):
        r = int(nxt())
        c = int(nxt())
        terminals.append((r, c))
    W = int(nxt())
    obstacles = set()
    for _ in range(W):
        r = int(nxt())
        c = int(nxt())
        obstacles.add((r, c))
    return R, C, terminals, obstacles


def main():
    if np is None:
        fail("numpy unavailable in checker environment")

    inp, outp = sys.argv[1], sys.argv[2]
    R, C, terminals, obstacles = read_instance(inp)
    node_id, coords, edges, term_nodes = pl.build_grid(R, C, obstacles, terminals)
    n = len(coords)
    m = len(edges)
    edge_index = pl.make_edge_index(coords, edges)
    pairs = pl.all_pairs(term_nodes)

    # --- parse participant output strictly ---
    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) < 2:
        fail("output too short: need mu then override-count")

    def parse_float(tok, name):
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric %s: %r" % (name, tok))
        if not math.isfinite(v):
            fail("non-finite %s: %r" % (name, tok))
        return v

    def parse_int(tok, name):
        try:
            return int(tok)
        except ValueError:
            fail("non-integer %s: %r" % (name, tok))

    mu = parse_float(raw[0], "mu")
    if not (pl.MU_MIN <= mu <= pl.MU_MAX):
        fail("mu=%.6f out of range [%.2f, %.2f]" % (mu, pl.MU_MIN, pl.MU_MAX))

    kov = parse_int(raw[1], "override count")
    if kov < 0 or kov > pl.MAX_OVERRIDES:
        fail("override count %d out of range [0,%d]" % (kov, pl.MAX_OVERRIDES))

    expected_tokens = 2 + 5 * kov
    if len(raw) != expected_tokens:
        fail("expected exactly %d tokens, got %d" % (expected_tokens, len(raw)))

    D0 = np.full(m, pl.D_BASE, dtype=float)
    p = 2
    for _ in range(kov):
        r1 = parse_int(raw[p], "r1"); c1 = parse_int(raw[p + 1], "c1")
        r2 = parse_int(raw[p + 2], "r2"); c2 = parse_int(raw[p + 3], "c2")
        d0 = parse_float(raw[p + 4], "d0")
        p += 5
        if not (0 <= r1 < R and 0 <= c1 < C):
            fail("override endpoint (%d,%d) out of grid bounds" % (r1, c1))
        if not (0 <= r2 < R and 0 <= c2 < C):
            fail("override endpoint (%d,%d) out of grid bounds" % (r2, c2))
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            fail("override endpoints (%d,%d)-(%d,%d) not orthogonally adjacent" % (r1, c1, r2, c2))
        if (r1, c1) not in node_id or (r2, c2) not in node_id:
            fail("override edge touches an obstacle cell")
        if not (pl.D_MIN <= d0 <= pl.D_MAX):
            fail("override d0=%.6f out of range [%.4f, %.2f]" % (d0, pl.D_MIN, pl.D_MAX))
        key = ((r1, c1), (r2, c2))
        if key not in edge_index:
            fail("override (%d,%d)-(%d,%d) is not a grid edge" % (r1, c1, r2, c2))
        D0[edge_index[key]] = d0

    # --- run the participant's simulation ---
    D_final = pl.run_sim_np(np, n, edges, pairs, D0, mu)
    net = pl.network_edges(edges, D_final)
    if not pl.bfs_reachable_all(n, net, term_nodes[0], term_nodes):
        fail("final tube network (D >= %.2f) does not connect all terminals" % pl.THRESH)
    F = pl.network_length(edges, D_final)
    if F <= 0:
        fail("empty tube network")

    # --- checker's own baseline: quiescent run, mu=1.0, no overrides ---
    D0_base = np.full(m, pl.D_BASE, dtype=float)
    D_base = pl.run_sim_np(np, n, edges, pairs, D0_base, 1.0)
    B = pl.network_length(edges, D_base)
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write(
        "mu=%.4f overrides=%d final_network_edges(F)=%d baseline(B)=%d\nRatio: %.6f\n"
        % (mu, kov, F, B, ratio)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
