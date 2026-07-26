# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    A, B, C, D, E = data[0:5]
    lines = []

    def emit(op, *args):
        lines.append(op + " " + " ".join(args))
        return "r%d" % len(lines)

    # Textbook direct transcription of the statement's formula. Looks minimal
    # and "obviously correct" -- and IS correct in double precision for
    # moderate x. It silently breaks at the extremes of the required domain:
    #   * sqrt(x^2+B^2) - x cancels catastrophically for large positive x
    #     (s rounds to exactly x, so the result rounds to exactly 0 even
    #     though the true value is a tiny nonzero residual).
    #   * log(1+exp(D*x)) overflows exp() for |D*x| beyond ~709.
    x2 = emit("MUL", "x", "x")
    b2 = emit("MUL", B, B)
    s2 = emit("ADD", x2, b2)
    s = emit("SQRT", s2)
    t1 = emit("SUB", s, "x")
    At1 = emit("MUL", A, t1)
    t2 = emit("DIV", B, s)
    Ct2 = emit("MUL", C, t2)
    dx = emit("MUL", D, "x")
    ex = emit("EXP", dx)
    op1 = emit("ADD", ex, "1.0")
    lg = emit("LOG", op1)
    Et3 = emit("MUL", E, lg)
    s1 = emit("ADD", At1, Ct2)
    f = emit("ADD", s1, Et3)

    out = [str(len(lines))] + lines
    print("\n".join(out))


if __name__ == "__main__":
    main()
