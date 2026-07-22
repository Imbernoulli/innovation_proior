import sys, random

# gen.py <testId>  --  "build the lock that only real keys visit"
#
# Family: dont-care-guided-synthesis.  We emit a PARTIAL truth table: a set of
# "real key" input vectors (the CARE set) each with a required output bit.
# Behaviour on every other input is FREE (don't-care).  The solver must output a
# straight-line GF(2) circuit (XOR/AND/OR/NOT/ONE/ZERO gates) that reproduces the
# required outputs on all care vectors, using AS FEW GATES AS POSSIBLE.
#
# PLANT (the whole point):
#   * The care set is exactly an affine subspace A = { R z + x0 : z_k..z_{n-1}=d }
#     of dimension k inside GF(2)^n  (the "real keys" all lie on one low-dim flat).
#   * On A, the target is a quadratic form of LOW symplectic rank 2*s':
#         g(z_0..z_{k-1}) = XOR_{p<s'} z_{2p} z_{2p+1}  XOR  <linear>  XOR  <const>
#     so its minimal multiplicative complexity is only s' AND gates.
#   * A random invertible linear map R mixes the coordinates, so in the RAW input
#     variables the SAME function looks like a DENSE, high-rank degree-2 polynomial
#     (many x_i x_j monomials).  An affine-oblivious solver that interpolates a
#     degree-2 form in the raw variables and emits one AND per quadratic monomial
#     (or even factors by a shared variable) spends FAR more AND gates than s'.
#
# TRAP: completing the don't-cares and synthesizing in the raw coordinates never
# sees the affine collapse; the insight is to detect A, change to free coordinates,
# and reduce the quadratic form to its low symplectic rank.
#
# Deterministic: everything seeded from testId only.

# (n, k, s_prime)  with 2*s_prime <= k < n,  m = 2^k care vectors.
SPECS = {
    1:  (14, 10, 3),
    2:  (15, 11, 3),
    3:  (16, 11, 2),
    4:  (16, 12, 3),
    5:  (17, 12, 2),
    6:  (18, 12, 3),
    7:  (18, 13, 2),
    8:  (20, 13, 2),
    9:  (22, 14, 2),
    10: (24, 14, 2),
}


def rand_invertible(rng, n):
    """Random invertible n x n matrix over GF(2), as list of column ints (bit i of
    col c = M[i][c]).  Returns (cols, ) ; retries until full rank."""
    while True:
        cols = [rng.getrandbits(n) for _ in range(n)]
        # rank check via elimination on a copy (treat cols as vectors)
        basis = []
        piv = {}
        ok = True
        for c in cols:
            v = c
            while v:
                b = v.bit_length() - 1
                if b in piv:
                    v ^= piv[b]
                else:
                    piv[b] = v
                    break
            if v == 0:
                ok = False
                break
        if ok and len(piv) == n:
            return cols


def main():
    tid = int(sys.argv[1])
    n, k, s_prime = SPECS[tid]
    rng = random.Random(20250539 + 100003 * tid)

    # --- affine map columns R (n ints, each an n-bit column) and offset x0 ---
    cols = rand_invertible(rng, n)
    x0 = rng.getrandbits(n)
    d = [rng.randint(0, 1) for _ in range(k, n)]  # fixed values of dependent coords

    # base point: x for z_free = 0  (dependent coords fixed to d)
    base = x0
    for idx, zi in enumerate(range(k, n)):
        if d[idx]:
            base ^= cols[zi]

    # planted function on free coords z_0..z_{k-1}:
    #   g(z) = XOR_{p<s'} A_p(z) * B_p(z)  XOR  Lin(z)  XOR  const
    # where A_p, B_p are 2*s' linearly-INDEPENDENT DENSE linear forms over the free
    # coords.  Low symplectic rank (2*s') -> s' AND gates suffice, BUT the quadratic
    # monomial graph is DENSE (each product ~ w^2 monomials), so a per-monomial or
    # shared-variable-factored circuit spends far more gates than the rank bound.
    def dense_form(nb):
        # a DENSE linear form over the k free coords (>= 2k/3 bits set) so each planted
        # product spans ~w^2 monomials; keeps the naive per-monomial baseline large.
        while True:
            v = rng.getrandbits(k)
            if bin(v).count("1") >= max(3, (2 * k) // 3):
                return v
    forms = []
    fb = []
    piv = {}
    while len(forms) < 2 * s_prime:
        v = dense_form(k)
        r = v
        for b in fb:
            lb = b.bit_length() - 1
            if (r >> lb) & 1:
                r ^= b
        if r:
            forms.append(v)
            # insert reduced into basis
            lb = r.bit_length() - 1
            fb.append(r)
    pairs = [(forms[2 * p], forms[2 * p + 1]) for p in range(s_prime)]
    lin_mask = rng.getrandbits(k)          # extra linear part over free coords
    const_bit = rng.randint(0, 1)

    def g_of_free(zf):
        val = const_bit
        for (Amask, Bmask) in pairs:
            a = bin(zf & Amask).count("1") & 1
            b = bin(zf & Bmask).count("1") & 1
            val ^= (a & b)
        val ^= bin(zf & lin_mask).count("1") & 1
        return val

    m = 1 << k
    rows = []
    for zf in range(m):
        x = base
        zz = zf
        i = 0
        while zz:
            if zz & 1:
                x ^= cols[i]
            zz >>= 1
            i += 1
        y = g_of_free(zf)
        rows.append((x, y))

    rng.shuffle(rows)

    out = ["%d %d" % (n, m)]
    for (x, y) in rows:
        bits = "".join("1" if (x >> i) & 1 else "0" for i in range(n))
        out.append("%s %d" % (bits, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
