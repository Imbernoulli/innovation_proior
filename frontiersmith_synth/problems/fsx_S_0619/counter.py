import sys

# Format-D checker for "cheapest polynomial multiplier chip".
# Instance:  "p d"  -> field F_p (p prime), two input polynomials of degree d.
# Artifact (stdout of the solver): a bilinear scheme (U, V, W) over F_p that
# computes the full product c(x) = a(x)*b(x) using r scalar products.
#   line 1 : r
#   r  rows of (d+1) ints : U  (left linear forms, one per product)
#   r  rows of (d+1) ints : V  (right linear forms, one per product)
#   (2d+1) rows of r ints : W  (recombination matrix -> output coefficients)
# We FIRST verify the exact bilinear identity, THEN score by product count r.
# Baseline B = schoolbook (d+1)^2 products.  ratio = min(1, 0.1 * B / r).


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    try:
        raw = open(inf).read().split()
        p = int(raw[0]); d = int(raw[1])
    except Exception:
        fail("badinstance"); return
    n = d + 1
    nc = 2 * d + 1

    try:
        toks = open(outf).read().split()
    except Exception:
        fail("noout"); return
    if not toks:
        fail("empty"); return

    try:
        r = int(toks[0])
    except ValueError:
        fail("r_parse"); return
    if r < 1 or r > 10 * n * n:
        fail("r_range"); return

    need = 1 + r * n * 2 + nc * r
    if len(toks) < need:
        fail("short"); return

    it = iter(toks[1:])

    def geti():
        v = int(next(it))  # rejects nan/inf/floats -> ValueError
        return v % p

    try:
        U = [[geti() for _ in range(n)] for _ in range(r)]
        V = [[geti() for _ in range(n)] for _ in range(r)]
        W = [[geti() for _ in range(r)] for _ in range(nc)]
    except (StopIteration, ValueError):
        fail("parse"); return

    # Verify the trilinear identity coefficient-by-coefficient:
    #   for all m,i,j :  sum_k W[m][k] U[k][i] V[k][j]  ==  [i + j == m]  (mod p)
    for m in range(nc):
        Wm = W[m]
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(r):
                    w = Wm[k]
                    if w:
                        s += w * U[k][i] * V[k][j]
                s %= p
                target = 1 if (i + j == m) else 0
                if s != target:
                    fail("identity m=%d i=%d j=%d" % (m, i, j)); return

    B = n * n
    sc = min(1000.0, 100.0 * B / max(1e-9, r))
    print("Ratio: %.6f" % (sc / 1000.0))


main()
