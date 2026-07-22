# TIER: strong
# The insight: every instance is secretly P(x) = g(x^2 + b*x + c) for some
# integer b,c (never given directly). "Completing the square" -- substituting
# x' = x + b/2 -- kills the linear term of the inner quadratic, so P becomes an
# EVEN polynomial in x'. That means:
#   1) b is recoverable directly from the top two coefficients: for n=2k,
#      a_{n-1} = a_n * k * b  (the x^{n-1} coefficient only ever gets a
#      contribution from the top block), so b = a_{n-1} / (a_n * k).
#   2) A Taylor shift of P by b/2 (an exact O(n^2) rational computation --
#      classic "derived-constant" preprocessing, done ONCE, off the runtime
#      path) produces the shifted coefficients; only the even-indexed ones are
#      nonzero, by construction.
#   3) At runtime: y = (x + b/2)^2 costs 1 multiplication, then Horner over
#      the k+1 shifted even coefficients costs k more -- k+1 = n/2+1 total,
#      versus Horner's n on the raw coefficients.
# If the planted structure is ever absent (defensive check fails), fall back
# to plain Horner so the program is always correct.
import sys
from fractions import Fraction as Fr


def poly_mul(A, B):
    C = [Fr(0)] * (len(A) + len(B) - 1)
    for i, av in enumerate(A):
        if av == 0:
            continue
        for j, bv in enumerate(B):
            if bv:
                C[i + j] += av * bv
    return C


def taylor_shift(a, shift):
    n = len(a) - 1
    Q = [Fr(0)] * (n + 1)
    base = [-shift, Fr(1)]      # (x' - shift)
    power = [Fr(1)]
    for i in range(n + 1):
        ai = a[i]
        if ai != 0:
            for j, coef in enumerate(power):
                Q[j] += ai * coef
        if i < n:
            power = poly_mul(power, base)
    return Q


def emit_const(lines, nxt_holder, val):
    lines.append("C %d %d" % (val.numerator, val.denominator))
    r = nxt_holder[0]
    nxt_holder[0] += 1
    return r


def emit(lines, nxt_holder, s):
    lines.append(s)
    r = nxt_holder[0]
    nxt_holder[0] += 1
    return r


def horner_fallback(a, n):
    lines = []
    nxt = [1]
    result = emit_const(lines, nxt, Fr(a[n]))
    for i in range(n - 1, -1, -1):
        mul_reg = emit(lines, nxt, "M %d 0" % result)
        if a[i] != 0:
            c_reg = emit_const(lines, nxt, Fr(a[i]))
            result = emit(lines, nxt, "A %d %d" % (mul_reg, c_reg))
        else:
            result = mul_reg
    return lines


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = [int(v) for v in data[1:1 + n + 1]]
    k = n // 2

    an = Fr(a[n])
    anm1 = Fr(a[n - 1])
    b = anm1 / (an * k)
    shift = b / 2

    Q = taylor_shift([Fr(v) for v in a], shift)
    ok = all(Q[i] == 0 for i in range(1, len(Q), 2))

    if not ok:
        lines = horner_fallback(a, n)
    else:
        A_list = [Q[2 * i] for i in range(k + 1)]
        lines = []
        nxt = [1]
        shift_reg = emit_const(lines, nxt, shift)
        xprime_reg = emit(lines, nxt, "A 0 %d" % shift_reg)
        y_reg = emit(lines, nxt, "M %d %d" % (xprime_reg, xprime_reg))

        result = emit_const(lines, nxt, A_list[k])
        for i in range(k - 1, -1, -1):
            mul_reg = emit(lines, nxt, "M %d %d" % (result, y_reg))
            if A_list[i] != 0:
                c_reg = emit_const(lines, nxt, A_list[i])
                result = emit(lines, nxt, "A %d %d" % (mul_reg, c_reg))
            else:
                result = mul_reg

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
