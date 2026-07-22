import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    tok = open(path).read().split()
    it = iter(tok)

    def nxt():
        return next(it)

    S = int(nxt()); I = int(nxt()); L = int(nxt()); M = int(nxt())
    cost = float(nxt()); cap = float(nxt())
    V = S + I + L
    nodedefs = []  # for node index S..V-1
    for _ in range(I + L):
        d = int(nxt())
        defs = []
        for _ in range(d):
            p = int(nxt())
            w = float(nxt())
            defs.append((p, w))
        nodedefs.append(defs)
    targets = []  # L x M
    for _ in range(L):
        targets.append([float(nxt()) for _ in range(M)])
    return dict(S=S, I=I, L=L, M=M, cost=cost, cap=cap, V=V,
               nodedefs=nodedefs, targets=targets)


def objective(inst, X):
    # X: S x M inlet concentrations already clamped to [0,cap]
    S, I, L, M, V = inst["S"], inst["I"], inst["L"], inst["M"], inst["V"]
    nodedefs, targets, cost = inst["nodedefs"], inst["targets"], inst["cost"]
    D = 0.0
    for s in range(M):
        val = [0.0] * V
        for j in range(S):
            val[j] = X[j][s]
        for idx, defs in enumerate(nodedefs):
            acc = 0.0
            for (p, w) in defs:
                acc += w * val[p]
            val[S + idx] = acc
        for l in range(L):
            leaf = S + I + l
            diff = val[leaf] - targets[l][s]
            D += diff * diff
    G = 0.0
    for j in range(S):
        for s in range(M):
            G += X[j][s]
    return D + cost * G


def main():
    inst = read_instance(sys.argv[1])
    S, M, cap = inst["S"], inst["M"], inst["cap"]

    out = open(sys.argv[2]).read().split()
    need = S * M
    if len(out) < need:
        fail("too few values (need %d got %d)" % (need, len(out)))

    X = [[0.0] * M for _ in range(S)]
    tol = 1e-6
    for j in range(S):
        for s in range(M):
            try:
                v = float(out[j * M + s])
            except Exception:
                fail("non-numeric value")
            if not math.isfinite(v):
                fail("non-finite value")
            if v < -tol or v > cap + tol:
                fail("out of range %g (cap %g)" % (v, cap))
            X[j][s] = min(cap, max(0.0, v))

    F = objective(inst, X)

    # internal baseline: all inlets at zero (inject nothing) -> F = sum targets^2
    B = 0.0
    for l in range(inst["L"]):
        for s in range(M):
            t = inst["targets"][l][s]
            B += t * t
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
