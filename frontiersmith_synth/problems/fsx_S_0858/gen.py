import sys

# gen.py <testId> -- prints ONE "masked bit-trick" black-box instance to stdout.
#
# The hidden function is f(x) = M2( T( M1(x) ) ) on w-bit words (mod 2^w arithmetic
# throughout), where:
#   M1(x) = ROTL(x, r1) XOR maskA        (affine "scramble in")
#   M2(z) = ROTL(z XOR maskB, r2)        (affine "scramble out")
#   T      = one of four fixed nonlinear word-tricks (see below), chosen per test.
# gen.py prints the COMPLETE truth table of f (all 2^w input->output pairs, in
# input order) -- the solver never has to guess w or query anything; it is handed
# the full black-box act and must find a SHORT program that reproduces it exactly.
#
# All ten cases use w=3 (8-row tables): the point of this family is that even a
# TINY table forces a real op-count gap between "reproduce the table" (linear in
# the table size) and "recognize the conjugated trick" (a small constant number of
# ops) -- brute-force construction cost still grows with table size while the
# insight-driven program does not, so the gap is genuine and not a size artifact.

W = 3
MASK = (1 << W) - 1


def rotl(v, r, w=W):
    r %= w
    if r == 0:
        return v & ((1 << w) - 1)
    m = (1 << w) - 1
    return ((v << r) | (v >> (w - r))) & m


def trick(y, tid, w=W):
    m = (1 << w) - 1
    if tid == 0:  # isolate lowest set bit
        return y & ((-y) & m)
    if tid == 1:  # smear the highest set bit downward (OR-reduce)
        z = y
        shift = 1
        while shift < w:
            z = z | (z >> shift)
            shift *= 2
        return z & m
    if tid == 2:  # parity broadcast (XOR-reduce then fill)
        z = y
        shift = 1
        while shift < w:
            z = z ^ (z >> shift)
            shift *= 2
        b = z & 1
        return (-b) & m
    if tid == 3:  # isolate lowest zero bit
        return ((~y) & m) & ((y + 1) & m)
    raise ValueError("bad trick id")


# Per-test-id plant: (trick_id, r1, maskA, r2, maskB)
# r1, r2 in [1, W-1] (nontrivial rotations); maskA, maskB in [0, 2^W - 1].
PLANTS = {
    1:  (0, 1, 0b011, 2, 0b101),
    2:  (1, 2, 0b110, 1, 0b010),
    3:  (2, 1, 0b001, 2, 0b111),
    4:  (3, 2, 0b100, 1, 0b011),
    5:  (0, 2, 0b111, 1, 0b001),
    6:  (1, 1, 0b010, 2, 0b101),
    7:  (2, 2, 0b011, 2, 0b110),
    8:  (3, 1, 0b101, 1, 0b010),
    9:  (0, 1, 0b110, 2, 0b011),
    10: (2, 2, 0b101, 1, 0b100),
}


def main():
    tid = int(sys.argv[1])
    tk, r1, maskA, r2, maskB = PLANTS[tid]
    n = 1 << W

    out = [str(W)]
    for x in range(n):
        y = rotl(x, r1) ^ maskA
        z = trick(y, tk, W)
        f = rotl(z ^ maskB, r2)
        out.append(str(f))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
