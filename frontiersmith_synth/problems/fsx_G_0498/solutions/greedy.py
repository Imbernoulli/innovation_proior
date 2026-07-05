# TIER: greedy
# Weight-1 syndrome decoder. Precomputes the syndrome of every single-bit error (the
# columns of H). For each frame: if its syndrome is zero it is already a codeword; if the
# syndrome matches a unique column, flip that one bit; otherwise give up (emit zeros).
# Corrects the weight-0 and weight-1 frames -> beats the trivial baseline.
import sys


def parity(x):
    return bin(x).count("1") & 1


def bits_to_int(s, n):
    v = 0
    for j, ch in enumerate(s):
        if ch == "1":
            v |= (1 << j)
    return v


def to_bits(v, n):
    return "".join("1" if (v >> j) & 1 else "0" for j in range(n))


def syndrome(H, x):
    s = 0
    for t, row in enumerate(H):
        if parity(row & x):
            s |= (1 << t)
    return s


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    while data[idx].strip() == "":
        idx += 1
    n, r, m, T = map(int, data[idx].split())
    idx += 1
    H = []
    for _ in range(r):
        while data[idx].strip() == "":
            idx += 1
        H.append(bits_to_int(data[idx].strip(), n))
        idx += 1

    # syndrome of a single flip at column j = j-th column of H
    col_synd = {}
    dup = set()
    for j in range(n):
        cs = 0
        for t, row in enumerate(H):
            if (row >> j) & 1:
                cs |= (1 << t)
        if cs in col_synd:
            dup.add(cs)
        col_synd[cs] = j
    for cs in dup:
        del col_synd[cs]

    out = []
    zero = to_bits(0, n)
    for _ in range(m):
        while data[idx].strip() == "":
            idx += 1
        y = bits_to_int(data[idx].strip(), n)
        idx += 1
        s = syndrome(H, y)
        if s == 0:
            out.append(to_bits(y, n))
        elif s in col_synd:
            out.append(to_bits(y ^ (1 << col_synd[s]), n))
        else:
            out.append(zero)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
