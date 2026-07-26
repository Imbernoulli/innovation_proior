# TIER: strong
import sys


def main():
    data = sys.stdin.read().split()
    A, B, C, D, E = data[0:5]
    lines = []

    def emit(op, *args):
        lines.append(op + " " + " ".join(args))
        return "r%d" % len(lines)

    # term1 = sqrt(x^2+B^2) - x, rewritten via the exact algebraic identity
    #   s - x  ==  (s - |x|) + (|x| - x)  ==  B^2/(s+|x|) + (|x| - x)
    # This single branchless form is uniformly stable: for x>=0 the
    # reciprocal avoids the direct s-x cancellation (denominator ~2x, large);
    # for x<0, |x|-x = 2|x| is an exact doubling (no cancellation at all) and
    # the reciprocal term is a genuinely tiny, harmless correction. One
    # identity, no regime test needed -- and it reuses s.
    x2 = emit("MUL", "x", "x")
    b2 = emit("MUL", B, B)
    s2 = emit("ADD", x2, b2)
    s = emit("SQRT", s2)
    ax = emit("ABS", "x")
    sax = emit("ADD", s, ax)
    rec = emit("DIV", b2, sax)
    axmx = emit("SUB", ax, "x")
    t1 = emit("ADD", rec, axmx)
    At1 = emit("MUL", A, t1)

    # term2 = B/s is already unconditionally stable and shares s with term1 --
    # the same "s" register that keeps term1 safe is exactly what term2 needs,
    # so accuracy and op-sharing come from the SAME reformulation, not two
    # separate passes.
    t2 = emit("DIV", B, s)
    Ct2 = emit("MUL", C, t2)

    # term3 = log(1+exp(D*x)), rewritten as the stable softplus split
    #   max(Dx,0) + log(1+exp(-|Dx|))
    # so the exp() argument is always <= 0 and can never overflow.
    dx = emit("MUL", D, "x")
    adx = emit("ABS", dx)
    ndx = emit("MUL", adx, "-1.0")
    ex = emit("EXP", ndx)
    op1 = emit("ADD", ex, "1.0")
    lg = emit("LOG", op1)
    sm = emit("ADD", dx, adx)
    mx = emit("MUL", sm, "0.5")
    t3 = emit("ADD", lg, mx)
    Et3 = emit("MUL", E, t3)

    s1 = emit("ADD", At1, Ct2)
    f = emit("ADD", s1, Et3)

    out = [str(len(lines))] + lines
    print("\n".join(out))


if __name__ == "__main__":
    main()
