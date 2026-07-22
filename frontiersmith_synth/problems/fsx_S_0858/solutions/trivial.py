# TIER: trivial
# Naive "one equality-indicator term per table row" construction. For every
# candidate value v it builds an exact isZero(x XOR v) flag from scratch (no
# sharing across rows) and accumulates v's table value when the flag fires.
# This is always correct for ANY table (it never looks at algebraic structure
# at all) but costs a fixed handful of ops per row -- exactly the checker's
# own baseline construction.
import sys


def main():
    data = sys.stdin.read().split()
    w = int(data[0])
    n = 1 << w
    table = [int(t) for t in data[1:1 + n]]

    lines = []

    def const(c):
        lines.append(("CONST", c % n))
        return len(lines)

    def op2(name, i, j):
        lines.append((name, i, j))
        return len(lines)

    def shift(name, i, c):
        lines.append((name, i, c))
        return len(lines)

    zero_r = const(0)
    one_r = const(1)
    acc = const(0)
    for v in range(n):
        v_r = const(v)
        d = op2("XOR", 0, v_r)          # x XOR v
        negd = op2("SUB", zero_r, d)    # -d (mod 2^w)
        t = op2("OR", d, negd)          # MSB set iff d != 0
        hi = shift("SHR", t, w - 1)     # 1 iff x != v
        iszero = op2("XOR", hi, one_r)  # 1 iff x == v
        f_r = const(table[v])
        contrib = op2("MUL", iszero, f_r)
        acc = op2("ADD", acc, contrib)

    out = [str(len(lines))]
    for ln in lines:
        out.append(" ".join(str(t) for t in ln))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
