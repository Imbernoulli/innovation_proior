# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); cap = int(next(it)); K = int(next(it))
    for _ in range(K):
        next(it); next(it)  # the textbook binary-splitting matrix never looks at
                             # the published double-fault list at all

    # Standard group-testing binary-splitting (counting-code) matrix: every
    # component gets a distinct 'bits'-wide binary address, one probe per bit
    # position. This is the classic recipe -- it perfectly separates every SINGLE
    # fault (unique codeword each) and stops there, spending no budget on
    # resolving which pairs of codewords happen to OR together into the same
    # observed pattern.
    bits = max(1, (n - 1).bit_length())
    rows = min(bits, m)

    out_rows = []
    for i in range(rows):
        bit = 1 << i
        row = ["1" if (j & bit) else "0" for j in range(n)]
        out_rows.append("".join(row))

    print(len(out_rows))
    for row in out_rows:
        print(row)


if __name__ == "__main__":
    main()
