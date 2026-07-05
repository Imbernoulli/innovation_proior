# TIER: strong
# Exploit the low real-rank structure with an EXACT alternating ternary rank-1 fit.
# For a residual R and fixed ternary v, the ternary u that maximizes the Frobenius
# reduction of subtracting one copy of u v^T is, per coordinate,
#     u_i = sign(s_i) if 2|s_i| > |v|^2 else 0 ,   s_i = <R_row_i, v> ,
# and symmetrically for v given u (all integer arithmetic). Alternate to a fixed
# point from several ternary seeds, peel the best-reducing term (repeatedly while it
# still helps), and finish any residual with the exact unary fallback. Return the
# fewest-term result among {peel, column-wise, row-wise}.
import sys

def read_M():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    M = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    return n, m, M

def sgn(x):
    return (x > 0) - (x < 0)

def rowwise(n, m, M):
    terms = []
    for i in range(n):
        h = max((abs(M[i][j]) for j in range(m)), default=0)
        for l in range(1, h + 1):
            u = [0] * n; u[i] = 1
            v = [sgn(M[i][j]) if abs(M[i][j]) >= l else 0 for j in range(m)]
            terms.append((u, v))
    return terms

def colwise(n, m, M):
    terms = []
    for j in range(m):
        h = max((abs(M[i][j]) for i in range(n)), default=0)
        for l in range(1, h + 1):
            v = [0] * m; v[j] = 1
            u = [sgn(M[i][j]) if abs(M[i][j]) >= l else 0 for i in range(n)]
            terms.append((u, v))
    return terms

def frob2(R, n, m):
    s = 0
    for i in range(n):
        ri = R[i]
        for j in range(m):
            s += ri[j] * ri[j]
    return s

def refine(R, n, m, v):
    # alternate ternary u <- v <- u to a fixed point
    for _ in range(30):
        vn = sum(x * x for x in v)
        if vn == 0:
            return None, None
        u = [0] * n
        for i in range(n):
            ri = R[i]; s = 0
            for j in range(m):
                if v[j]:
                    s += ri[j] * v[j]
            if 2 * abs(s) > vn:
                u[i] = 1 if s > 0 else -1
        un = sum(x * x for x in u)
        if un == 0:
            return None, None
        nv = [0] * m
        for j in range(m):
            s = 0
            for i in range(n):
                ui = u[i]
                if ui:
                    s += R[i][j] * ui
            if 2 * abs(s) > un:
                nv[j] = 1 if s > 0 else -1
        if nv == v:
            v = nv
            break
        v = nv
    un = sum(x * x for x in u)
    vn = sum(x * x for x in v)
    if un == 0 or vn == 0:
        return None, None
    return u, v

def reduction(R, u, v, n, m):
    un = sum(x * x for x in u)
    vn = sum(x * x for x in v)
    dot = 0
    for i in range(n):
        ui = u[i]
        if ui:
            ri = R[i]
            for j in range(m):
                if v[j]:
                    dot += ui * v[j] * ri[j]
    return 2 * dot - un * vn

def subtract(R, u, v, n, m):
    for i in range(n):
        ui = u[i]
        if ui:
            ri = R[i]
            for j in range(m):
                if v[j]:
                    ri[j] -= ui * v[j]

def peel(n, m, M):
    R = [row[:] for row in M]
    terms = []
    guard = 0
    limit = 4 * (n + m + sum(abs(M[i][j]) for i in range(n) for j in range(m)) + 1)
    while frob2(R, n, m) > 0 and guard < limit:
        guard += 1
        # seeds: ternary sign of each residual row + each residual column
        seeds = []
        for i in range(n):
            row = R[i]
            if any(row):
                seeds.append([sgn(x) for x in row])
        for j in range(m):
            col = [R[i][j] for i in range(n)]
            if any(col):
                # this seed is length-n; use it as u-seed by transposing: build v from it
                pass
        best = None; best_red = 0
        seen = set()
        for s in seeds:
            key = tuple(s)
            if key in seen:
                continue
            seen.add(key)
            u, v = refine(R, n, m, s)
            if u is None:
                continue
            red = reduction(R, u, v, n, m)
            if red > best_red:
                best_red = red; best = (u, v)
        if best is None or best_red <= 0:
            break
        u, v = best
        # peel this term repeatedly while a single copy still reduces the residual
        while reduction(R, u, v, n, m) > 0:
            subtract(R, u, v, n, m)
            terms.append((u[:], v[:]))
    # exact fallback on any remaining residual
    terms += rowwise(n, m, R)
    return terms

def emit(terms, n, m):
    if not terms:
        terms = [([1] + [0] * (n - 1), [0] * m)]
    outl = [str(len(terms))]
    for u, v in terms:
        outl.append(" ".join(map(str, u)))
        outl.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(outl) + "\n")

def main():
    n, m, M = read_M()
    cands = [rowwise(n, m, M), colwise(n, m, M), peel(n, m, M)]
    best = min(cands, key=len)
    emit(best, n, m)

if __name__ == "__main__":
    main()
