# TIER: strong
# The insight: DON'T synthesize the table row by row. First hypothesize that
# the black box is a CONJUGATED IDENTITY f = M2 o T o M1 with M1(x) =
# ROTL(x,r1) XOR maskA, M2(z) = ROTL(z XOR maskB, r2), and T drawn from a
# small catalogue of classic word-tricks (isolate-lowest-set-bit, smear,
# parity-broadcast, isolate-lowest-zero-bit). That hypothesis turns "search
# over all straight-line programs" into "search over five small algebraic
# parameters" (trick id, r1, maskA, r2, maskB) -- a superoptimization search
# restricted by the recognized conjugated-identity structure. Once the
# parameters are pinned down (checked against the FULL supplied table, so the
# match is exact, not a guess) we emit the tiny fixed-shape program for that
# hypothesis: 4 ops to undo/redo each affine wrapper plus the trick's own
# small op count -- a handful of ops total, independent of the table size,
# instead of one term per row.
import sys


def rotl(v, r, w):
    r %= w
    m = (1 << w) - 1
    if r == 0:
        return v & m
    return ((v << r) | (v >> (w - r))) & m


def trick_val(y, tid, w):
    m = (1 << w) - 1
    if tid == 0:
        return y & ((-y) & m)
    if tid == 1:
        z = y
        s = 1
        while s < w:
            z = z | (z >> s)
            s *= 2
        return z & m
    if tid == 2:
        z = y
        s = 1
        while s < w:
            z = z ^ (z >> s)
            s *= 2
        b = z & 1
        return (-b) & m
    if tid == 3:
        return ((~y) & m) & ((y + 1) & m)
    raise ValueError


def find_plant(w, table):
    n = 1 << w
    for tid in range(4):
        for r1 in range(1, w):
            for r2 in range(1, w):
                for maskA in range(n):
                    for maskB in range(n):
                        ok = True
                        for x in range(n):
                            y = rotl(x, r1, w) ^ maskA
                            z = trick_val(y, tid, w)
                            f = rotl(z ^ maskB, r2, w)
                            if f != table[x]:
                                ok = False
                                break
                        if ok:
                            return tid, r1, maskA, r2, maskB
    return None


def main():
    data = sys.stdin.read().split()
    w = int(data[0])
    n = 1 << w
    table = [int(t) for t in data[1:1 + n]]

    plant = find_plant(w, table)
    lines = []

    def const(c):
        lines.append(("CONST", c % n))
        return len(lines)

    def op2(name, i, j):
        lines.append((name, i, j))
        return len(lines)

    def op1(name, i):
        lines.append((name, i))
        return len(lines)

    def shift(name, i, c):
        lines.append((name, i, c))
        return len(lines)

    if plant is not None:
        tid, r1, maskA, r2, maskB = plant

        # M1(x) = ROTL(x, r1) XOR maskA
        s1 = shift("SHL", 0, r1)
        s2 = shift("SHR", 0, w - r1)
        rot1 = op2("OR", s1, s2)
        maskA_r = const(maskA)
        y = op2("XOR", rot1, maskA_r)

        # T(y)
        if tid == 0:
            zero_r = const(0)
            negy = op2("SUB", zero_r, y)
            z = op2("AND", y, negy)
        elif tid == 1:
            cur = y
            s = 1
            while s < w:
                sh = shift("SHR", cur, s)
                cur = op2("OR", cur, sh)
                s *= 2
            z = cur
        elif tid == 2:
            cur = y
            s = 1
            while s < w:
                sh = shift("SHR", cur, s)
                cur = op2("XOR", cur, sh)
                s *= 2
            one_r = const(1)
            b = op2("AND", cur, one_r)
            zero_r = const(0)
            z = op2("SUB", zero_r, b)
        else:  # tid == 3
            noty = op1("NOT", y)
            one_r2 = const(1)
            yp1 = op2("ADD", y, one_r2)
            z = op2("AND", noty, yp1)

        # M2(z) = ROTL(z XOR maskB, r2)
        maskB_r = const(maskB)
        zx = op2("XOR", z, maskB_r)
        t1 = shift("SHL", zx, r2)
        t2 = shift("SHR", zx, w - r2)
        op2("OR", t1, t2)
    else:
        # Should not happen for the planted family, but stay correct: fall
        # back to the same universally-correct indicator-sum as trivial.
        zero_r = const(0)
        one_r = const(1)
        acc = const(0)
        for v in range(n):
            v_r = const(v)
            d = op2("XOR", 0, v_r)
            negd = op2("SUB", zero_r, d)
            tt = op2("OR", d, negd)
            hi = shift("SHR", tt, w - 1)
            iszero = op2("XOR", hi, one_r)
            f_r = const(table[v])
            contrib = op2("MUL", iszero, f_r)
            acc = op2("ADD", acc, contrib)

    out = [str(len(lines))]
    for ln in lines:
        out.append(" ".join(str(t) for t in ln))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
