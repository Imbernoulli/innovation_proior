import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def clearing(pbar, incoming, e_ext, max_iters):
    """Greatest Eisenberg-Noe clearing payment vector via the standard
    monotone-decreasing fixed-point iteration p <- min(pbar, e + Pi^T p),
    starting from p = pbar. incoming[i] = list of (creditor j, weight w)
    meaning bank j owes bank i amount w (so bank i's inflow includes
    (w / pbar[j]) * p[j])."""
    n = len(pbar)
    p = list(pbar)
    for _ in range(max_iters):
        changed = False
        newp = [0.0] * n
        for i in range(n):
            pb = pbar[i]
            if pb <= 0:
                newp[i] = 0.0
                continue
            inflow = 0.0
            for (j, w) in incoming[i]:
                pbj = pbar[j]
                if pbj > 0:
                    inflow += (w / pbj) * p[j]
            val = e_ext[i] + inflow
            pi_new = pb if val >= pb else val
            newp[i] = pi_new
            if abs(pi_new - p[i]) > 1e-9:
                changed = True
        p = newp
        if not changed:
            break
    return p


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read files")

    try:
        it = iter(inp)
        N = int(next(it)); C = int(next(it))
        if N <= 0 or C < 0:
            fail("bad N/C")
        e = [float(next(it)) for _ in range(N)]
        M = int(next(it))
        pbar = [0.0] * N
        incoming = [[] for _ in range(N)]
        seen_edges = set()
        for _ in range(M):
            u = int(next(it)) - 1
            v = int(next(it)) - 1
            w = float(next(it))
            if not (0 <= u < N and 0 <= v < N) or u == v or w < 0:
                fail("bad edge")
            if (u, v) in seen_edges:
                fail("duplicate edge")
            seen_edges.add((u, v))
            pbar[u] += w
            incoming[v].append((u, w))
    except Exception:
        fail("bad input")

    max_iters = N + 5

    # ---- internal baseline B: do nothing (delta = 0) ----
    p0 = clearing(pbar, incoming, e, max_iters)
    B = sum(pbar[i] - p0[i] for i in range(N))
    B = max(B, 1e-6)

    # ---- parse & validate participant output ----
    if len(out_toks) != N:
        fail("expected %d numbers, got %d" % (N, len(out_toks)))
    delta = []
    for tok in out_toks:
        try:
            d = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(d):
            fail("non-finite value")
        if d < -1e-6:
            fail("negative injection")
        delta.append(max(0.0, d))

    tot = sum(delta)
    if tot > C + 1e-4:
        fail("budget exceeded: sum=%.4f > C=%d" % (tot, C))

    e_prime = [e[i] + delta[i] for i in range(N)]
    p1 = clearing(pbar, incoming, e_prime, max_iters)
    F = sum(pbar[i] - p1[i] for i in range(N))
    F = max(F, 0.0)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
