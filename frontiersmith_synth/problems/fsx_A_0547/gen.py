import sys, random

# reagent-mixing-manifold : a fixed DAG "mixing manifold".
#   nodes 0..S-1        = inlet tanks (sources, no parents)
#   nodes S..S+I-1      = internal mixers (weighted average of parents)
#   nodes S+I..S+I+L-1  = leaf outputs (also weighted averages; no children)
# A few "trunk" inlets are sampled far more often as parents, so they feed many
# leaves -> ill-conditioned, coupled columns (the bottleneck structure).
# Targets are built from an OUT-OF-BOX reference vector x_ref so the exact match
# is unreachable: the achievable leaf set is the zonotope {A x : 0<=x<=1}, and the
# solver must PROJECT onto it (not invert-then-clip).


def gen_case(i):
    rng = random.Random(540700 + 97 * i)

    # difficulty ladder
    if i <= 3:
        S, I, L, M = 5, 8, 5, 2
    elif i <= 7:
        S, I, L, M = 12, 34, 12, 4
    else:
        S, I, L, M = 22, 96, 22, 6

    cost = 0.02
    ntrunk = 2 if S <= 5 else 3

    # trap cases: heavier out-of-box excursion concentrated on trunk inlets
    trap = i in (3, 6, 9, 10)

    V = S + I + L
    trunk = list(range(ntrunk))  # inlets 0..ntrunk-1 are bottlenecks

    # ---- build DAG node definitions for nodes S..V-1 ----
    nodedefs = []  # list of (list of (parent, weight))
    for j in range(S, V):
        cand = list(range(j))  # any earlier node
        # sampling weights: trunk inlets favored strongly so influence spreads wide
        prob = []
        for c in cand:
            if c in trunk:
                prob.append(6.0)
            elif c < S:
                prob.append(1.0)
            else:
                prob.append(2.0)
        d = rng.randint(2, 3)
        d = min(d, len(cand))
        # weighted sampling without replacement
        chosen = []
        pool = cand[:]
        pr = prob[:]
        for _ in range(d):
            tot = sum(pr)
            r = rng.random() * tot
            acc = 0.0
            pick = 0
            for idx in range(len(pool)):
                acc += pr[idx]
                if r <= acc:
                    pick = idx
                    break
            chosen.append(pool[pick])
            del pool[pick]
            del pr[pick]
        # random positive weights, normalized to sum 1
        raw = [0.2 + rng.random() for _ in chosen]
        tot = sum(raw)
        w = [x / tot for x in raw]
        nodedefs.append(list(zip(chosen, w)))

    # ---- forward influence matrix A (L x S) to build unreachable targets ----
    # coef[node] = length-S vector of inlet contributions
    coef = [[0.0] * S for _ in range(V)]
    for j in range(S):
        coef[j][j] = 1.0
    for idx, defs in enumerate(nodedefs):
        j = S + idx
        row = coef[j]
        for (p, w) in defs:
            cp = coef[p]
            for s in range(S):
                row[s] += w * cp[s]
    A = [coef[S + I + l] for l in range(L)]  # L x S

    # ---- reference inlet vector x_ref (out of box) per species -> targets ----
    # x_ref lives well OUTSIDE [0,1], so the exact match is unreachable: the leaf
    # outputs A x_ref leave the achievable zonotope, leaving irreducible residual
    # (keeps score headroom) and forcing the unconstrained inverse out of the box.
    lo, hi = (-1.05, 2.05) if trap else (-0.7, 1.7)
    targets = [[0.0] * M for _ in range(L)]
    for s in range(M):
        xref = [lo + (hi - lo) * rng.random() for _ in range(S)]
        # push trunk inlets to extremes so unconstrained-invert blows up on them
        for t in trunk:
            ext = 0.6 if trap else 0.35
            xref[t] = hi + ext if rng.random() < 0.5 else lo - ext
        for l in range(L):
            targets[l][s] = sum(A[l][s2] * xref[s2] for s2 in range(S))

    # ---- emit ----
    out = []
    out.append("%d %d %d %d %s 1.0" % (S, I, L, M, repr(cost)))
    for defs in nodedefs:
        parts = [str(len(defs))]
        for (p, w) in defs:
            parts.append(str(p))
            parts.append(repr(w))
        out.append(" ".join(parts))
    for l in range(L):
        out.append(" ".join(repr(targets[l][s]) for s in range(M)))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    i = int(sys.argv[1])
    gen_case(i)


if __name__ == "__main__":
    main()
