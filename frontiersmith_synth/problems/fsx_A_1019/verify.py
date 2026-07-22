#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Relief Bazaar problem.

Reads the instance, parses the participant's INITIAL allocation strictly, replays the
fixed friction-limited bilateral-trade protocol (community-local only) for R rounds,
scores the resulting Nash Social Welfare F, compares it against the Nash Social
Welfare B of the same replay applied to an even/proportional reference split, and
prints  Ratio: <F/(10B) capped at 1.0>.
"""
import sys
import re
import math

INT_RE = re.compile(r'^[+-]?\d+$')
FLOOR = 1  # subsistence utility with zero relief goods


def fail(msg):
    print("INFEASIBLE: %s  Ratio: 0.0" % msg)
    sys.exit(0)


def parse_instance(path):
    try:
        toks = open(path).read().split()
    except Exception as e:
        fail("cannot read instance: %s" % e)
    idx = [0]

    def nxt():
        v = toks[idx[0]]
        idx[0] += 1
        return v

    try:
        N = int(nxt()); G = int(nxt()); K = int(nxt()); R = int(nxt()); eps = int(nxt())
        cap = [int(nxt()) for _ in range(G)]
        S = [int(nxt()) for _ in range(G)]
        comm = [0] * N
        W = [[0] * G for _ in range(N)]
        for i in range(N):
            comm[i] = int(nxt())
            for g in range(G):
                W[i][g] = int(nxt())
    except Exception as e:
        fail("malformed instance: %s" % e)
    return N, G, K, R, eps, cap, S, comm, W


def parse_allocation(path, N, G, S):
    try:
        txt = open(path).read()
    except Exception as e:
        fail("cannot read output: %s" % e)
    toks = txt.split()
    if len(toks) != N * G:
        fail("expected %d integer tokens, got %d" % (N * G, len(toks)))
    x = [[0] * G for _ in range(N)]
    k = 0
    for i in range(N):
        for g in range(G):
            t = toks[k]; k += 1
            if not INT_RE.match(t):
                fail("non-integer/non-finite token '%s'" % t[:20])
            v = int(t)
            if v < 0:
                fail("negative allocation at (%d,%d)" % (i, g))
            x[i][g] = v
    for g in range(G):
        col = sum(x[i][g] for i in range(N))
        if col != S[g]:
            fail("good %d supply mismatch: got %d, need %d" % (g, col, S[g]))
    return x


REF_FRAC = 0.85  # reference batch covers this fraction of the roster


def equal_split(N, G, S):
    """The checker's own trivial reference: split each good evenly among the
    first ROUND(REF_FRAC * N) households by index (a fixed-size first batch),
    leaving the remaining households with none."""
    M = max(1, round(REF_FRAC * N))
    x = [[0] * G for _ in range(N)]
    for g in range(G):
        base, rem = S[g] // M, S[g] % M
        for i in range(M):
            x[i][g] = base + (1 if i < rem else 0)
    return x


def replay(x0, N, G, comm, W, cap, R, eps):
    x = [row[:] for row in x0]
    groups = {}
    for i in range(N):
        groups.setdefault(comm[i], []).append(i)
    order = []
    for c in sorted(groups):
        members = groups[c]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                order.append((members[a], members[b]))
    for _ in range(R):
        changed = False
        for (i, j) in order:
            xi, xj = x[i], x[j]
            wi, wj = W[i], W[j]
            for g in range(G):
                if xi[g] <= 0:
                    continue
                for h in range(G):
                    if h == g or xj[h] <= 0:
                        continue
                    gain_i = wi[h] if xi[h] < cap[h] else 0
                    loss_i = wi[g] if xi[g] <= cap[g] else 0
                    dUi = gain_i - loss_i
                    if dUi < eps:
                        continue
                    gain_j = wj[g] if xj[g] < cap[g] else 0
                    loss_j = wj[h] if xj[h] <= cap[h] else 0
                    dUj = gain_j - loss_j
                    if dUj < eps:
                        continue
                    xi[g] -= 1; xi[h] += 1
                    xj[h] -= 1; xj[g] += 1
                    changed = True
        if not changed:
            break
    return x


def nsw(x, N, G, W, cap):
    total_log = 0.0
    for i in range(N):
        u = FLOOR
        for g in range(G):
            k = x[i][g]
            if k > cap[g]:
                k = cap[g]
            if k > 0:
                u += W[i][g] * k
        total_log += math.log(u)
    return math.exp(total_log / N)


def main():
    inpath, outpath = sys.argv[1], sys.argv[2]
    N, G, K, R, eps, cap, S, comm, W = parse_instance(inpath)
    x = parse_allocation(outpath, N, G, S)

    xf = replay(x, N, G, comm, W, cap, R, eps)
    F = nsw(xf, N, G, W, cap)

    x0 = equal_split(N, G, S)
    xb = replay(x0, N, G, comm, W, cap, R, eps)
    B = nsw(xb, N, G, W, cap)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F: %.6f  B: %.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
