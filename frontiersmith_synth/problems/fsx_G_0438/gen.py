import sys

# gen.py <testId>  -- prints ONE fixed filter-polynomial evaluation instance.
#
# DSP setting: a linear-phase FIR / all-zero filter stage is characterised by a
# transfer-function numerator H(z) with fixed integer taps.  In the pipeline this
# polynomial must be evaluated at a symbolic sample value using as FEW scalar
# MULTIPLICATIONS as possible (multiplies are the expensive MAC resource on a DSP
# core / FPGA; additions are cheap).
#
# Each planted H is a MONIC polynomial of degree d = 2k that happens to equal a
# single quadratic section raised to the k-th power:
#            H(x) = (x^2 + p*x + q)^k        (p integer, q >= 1 integer)
# The solver only ever sees the DENSE expanded integer tap vector -- the factored
# structure is hidden.  Recognising the structure and then scheduling the powers
# with a short addition chain is the open-ended task; Horner (d mults) is the
# naive baseline and the true minimum is unknown (shortest addition chains).
#
# Difficulty grows with testId (larger k, more taps).

# testId -> (k, p, q).  q kept positive so the constant tap q^k has a unique
# real k-th root (clean, but the solver still must DISCOVER it from the taps).
SPECS = {
    1:  (4,   1, 1),
    2:  (6,   2, 2),
    3:  (8,   3, 1),
    4:  (10,  1, 3),
    5:  (12,  2, 2),
    6:  (16,  3, 1),
    7:  (20, -1, 2),
    8:  (24, -2, 3),
    9:  (28, -3, 1),
    10: (31,  2, 2),
}


def poly_pow(p, q, k):
    # exact integer coefficients of (x^2 + p*x + q)^k, index i = coeff of x^i.
    base = [q, p, 1]              # x^2 + p x + q
    res = [1]                    # the constant 1
    for _ in range(k):
        out = [0] * (len(res) + len(base) - 1)
        for i, a in enumerate(res):
            if a == 0:
                continue
            for j, b in enumerate(base):
                out[i + j] += a * b
        res = out
    return res


def main():
    tid = int(sys.argv[1])
    if tid not in SPECS:
        tid = ((tid - 1) % 10) + 1
    k, p, q = SPECS[tid]
    coeffs = poly_pow(p, q, k)   # length 2k+1
    d = len(coeffs) - 1
    out = []
    out.append(str(d))
    out.append(" ".join(str(c) for c in coeffs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
