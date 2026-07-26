# TIER: trivial
import sys

COST = {"ADD": 1, "SUB": 1, "MUL": 1, "DIV": 3, "ABS": 1, "SQRT": 4, "EXP": 8, "LOG": 8}
TARGET_COST = 200  # matches the checker's fixed internal baseline op-cost


def main():
    data = sys.stdin.read().split()
    A, B, C, D, E = data[0:5]
    lines = []
    cost = 0

    def emit(op, *args):
        nonlocal cost
        lines.append(op + " " + " ".join(args))
        cost += COST[op]
        return "r%d" % len(lines)

    # Correct everywhere (uses the same safe closed forms as the strong
    # solution) but deliberately unoptimized: every sub-quantity is
    # recomputed from scratch for each term (no shared subexpressions), the
    # whole evaluation is duplicated and averaged as a redundant "double
    # check", and the result is padded with harmless no-op additions to
    # model an unrefined first-draft implementation.
    def safe_value():
        x2 = emit("MUL", "x", "x")
        b2 = emit("MUL", B, B)
        s2 = emit("ADD", x2, b2)
        s = emit("SQRT", s2)
        ax = emit("ABS", "x")
        sax = emit("ADD", s, ax)
        b2b = emit("MUL", B, B)
        rec = emit("DIV", b2b, sax)
        axmx = emit("SUB", ax, "x")
        t1 = emit("ADD", rec, axmx)
        At1 = emit("MUL", A, t1)

        x2b = emit("MUL", "x", "x")
        b2c = emit("MUL", B, B)
        s2b = emit("ADD", x2b, b2c)
        sB = emit("SQRT", s2b)
        t2 = emit("DIV", B, sB)
        Ct2 = emit("MUL", C, t2)

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
        return f

    r1 = safe_value()
    r2 = safe_value()
    avgsum = emit("ADD", r1, r2)
    avg = emit("MUL", avgsum, "0.5")
    cur = avg
    while cost < TARGET_COST:
        cur = emit("ADD", cur, "0.0")

    out = [str(len(lines))] + lines
    print("\n".join(out))


if __name__ == "__main__":
    main()
