# TIER: greedy
"""The 'obvious' first idea: cluster states by their next-state under a SINGLE fixed
symbol (symbol 0), sort states by that one feature, then lay the sorted order out with
a standard reflected-binary Gray code (so adjacent sorted positions get Hamming-distance-1
codewords). This captures a little adjacency but only looks at 1 of the M symbols, so it
misses any structure that only shows up through the other symbols (and is fully blind on
the 'trap' instances where symbol 0's transitions are state-independent)."""
import sys


def gray_code(p, B):
    g = p ^ (p >> 1)
    return format(g, f"0{B}b")


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    M = 1 << K
    B = N.bit_length() - 1
    trans = []
    for _u in range(N):
        row = [int(next(it)) for _ in range(M)]
        trans.append(row)

    order = sorted(range(N), key=lambda u: (trans[u][0], u))

    codeword = [None] * N
    for pos, u in enumerate(order):
        codeword[u] = gray_code(pos, B)

    sys.stdout.write("\n".join(codeword) + "\n")


if __name__ == "__main__":
    main()
