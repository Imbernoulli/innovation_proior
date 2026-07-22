# TIER: greedy
# Textbook, structure-oblivious binary decision diagram (multiplexer tree).
# Precompute one branch-free "broadcast(bit_i(x))" mask per input bit, then
# fold the table bottom-up with select(bit, hi, lo) = lo XOR (mask AND (hi
# XOR lo)) muxes. This is exactly correct for ANY table by construction (it
# never asks WHY the table looks the way it does) and roughly halves the
# naive per-row cost via node sharing -- a real, valid improvement any
# competent coder reaches for, but it still touches every row of the table
# and never notices the conjugated M2(T(M1(x))) shape.
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

    bcast = []
    for i in range(w):
        bi = shift("SHR", 0, i)
        bit = op2("AND", bi, one_r)
        bc = op2("SUB", zero_r, bit)  # all-ones if bit i of x set, else 0
        bcast.append(bc)

    leaves = [const(table[v]) for v in range(n)]

    def build(sub, bitpos):
        if bitpos < 0:
            return sub[0]
        half = len(sub) // 2
        lo = build(sub[:half], bitpos - 1)
        hi = build(sub[half:], bitpos - 1)
        diff = op2("XOR", hi, lo)
        masked = op2("AND", diff, bcast[bitpos])
        return op2("XOR", lo, masked)

    build(leaves, w - 1)

    out = [str(len(lines))]
    for ln in lines:
        out.append(" ".join(str(t) for t in ln))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
