# TIER: greedy
# "This is textbook Ben-Or-Tiwari / Prony": treat the t known-sparse terms as an
# order-t linear recurrence and fit it from samples s_0..s_{2t-1} (the standard,
# structure-UNAWARE recipe -- ignores that exponents are said to form an AP).
# Builds the order-t Toeplitz system directly from the given samples; if the
# query budget Q is below the textbook requirement 2t, the missing future
# samples are silently treated as 0 (a realistic "just use what indices exist"
# bug), which corrupts the recurrence and the recovered term list on TIGHT
# instances. On LOOSE instances (Q >= 2t) this recipe has all the data it needs
# and recovers the seam exactly, same as an insightful solver.
import sys, json


def gauss_solve(A, b, p, n):
    A = [row[:] for row in A]
    b = b[:]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if A[r][col] % p != 0:
                piv = r; break
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = pow(A[col][col], p - 2, p)
        A[col] = [(x * inv) % p for x in A[col]]
        b[col] = (b[col] * inv) % p
        for r in range(n):
            if r != col and A[r][col] % p != 0:
                factor = A[r][col] % p
                A[r] = [(A[r][j] - factor * A[col][j]) % p for j in range(n)]
                b[r] = (b[r] - factor * b[col]) % p
    return [x % p for x in b]


inst = json.load(sys.stdin)
p = inst["p"]; g = inst["g"]; t = inst["t"]; Q = inst["Q"]; s = inst["s"]

m = 2 * t
sx = [(s[k] if k < Q else 0) for k in range(m)]   # zero-pad missing future samples

M = [[sx[i + j] % p for j in range(t)] for i in range(t)]
rhs = [sx[i + t] % p for i in range(t)]
lam = gauss_solve(M, rhs, p, t)

terms = []
if lam is not None:
    # roots of char poly X^t - sum_j lam_j X^j over F_p (brute force; p is small)
    roots = []
    for x in range(1, p):
        val = pow(x, t, p)
        for j in range(t):
            val = (val - lam[j] * pow(x, j, p)) % p
        if val % p == 0:
            roots.append(x)
            if len(roots) == t:
                break
    if roots:
        tt = len(roots)
        M2 = [[pow(roots[i], k, p) for i in range(tt)] for k in range(tt)]
        rhs2 = [s[k] % p if k < Q else 0 for k in range(tt)]
        c2 = gauss_solve(M2, rhs2, p, tt)
        if c2 is not None:
            # discrete log each root back to an exponent (brute force table; p small)
            dlog = {}
            acc = 1
            for e in range(p - 1):
                dlog.setdefault(acc, e)
                acc = (acc * g) % p
            for i in range(tt):
                if roots[i] in dlog:
                    terms.append([dlog[roots[i]], c2[i]])

print(json.dumps({"terms": terms[:t]}))
