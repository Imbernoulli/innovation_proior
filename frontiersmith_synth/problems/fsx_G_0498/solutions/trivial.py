# TIER: trivial
# Do-nothing decoder: if a received frame is already a codeword, echo it; otherwise emit
# the all-zero codeword. This corrects exactly the clean (weight-0) frames -- the same
# set the checker uses as its baseline B -- so it scores Ratio ~= 0.1.
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
    out = []
    zero = to_bits(0, n)
    for _ in range(m):
        while data[idx].strip() == "":
            idx += 1
        y = bits_to_int(data[idx].strip(), n)
        idx += 1
        if all(not parity(row & y) for row in H):
            out.append(to_bits(y, n))
        else:
            out.append(zero)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
