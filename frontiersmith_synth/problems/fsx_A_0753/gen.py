import sys

# Difficulty/trap ladder. Each row: (L, K, Bmin, Bmax, G, pairs)
#   L    = codeword length
#   K    = codebook size (number of codewords)
#   Bmin,Bmax = single-burst length range swept at every cyclic offset
#   G    = gap between the two sub-bursts of a composite double-burst
#   pairs = list of (b1,b2) composite double-burst length pairs swept at every offset
#
# Cases 2,4,6,8 are engineered TRAPS: Bmax is deliberately > (L // ceil(log2 K)) / 2,
# i.e. bigger than half a "naive equal-partition block" length, so a code that just
# dedicates one contiguous block of bits per message-index-bit (the obvious
# max-pairwise-Hamming-distance recipe) gets a single swept burst fully swallowed
# inside one block, wiping out its decode margin. Cases 1,3,5,7,9,10 use a mild
# burst range that does not trigger this.
CASES = [
    (42,  6, 2, 3,  2, [(2, 2)]),    # 1  non-trap, small
    (45,  6, 4, 8,  3, [(2, 3)]),    # 2  TRAP
    (56,  8, 2, 3,  2, [(2, 2)]),    # 3  non-trap
    (76,  9, 6, 10, 3, [(3, 4)]),    # 4  TRAP
    (70, 10, 2, 3,  2, [(2, 2)]),    # 5  non-trap
    (76, 10, 6, 10, 3, [(3, 4)]),    # 6  TRAP
    (80, 11, 2, 3,  2, [(2, 2)]),    # 7  non-trap
    (76, 12, 6, 10, 3, [(3, 4)]),    # 8  TRAP
    (84, 12, 2, 3,  2, [(2, 2)]),    # 9  non-trap
    (112,16, 2, 3,  2, [(2, 2)]),    # 10 non-trap, largest
]


def main():
    i = int(sys.argv[1])
    i = max(1, min(len(CASES), i))
    L, K, Bmin, Bmax, G, pairs = CASES[i - 1]

    out = [f"{L} {K}", f"{Bmin} {Bmax}", f"{G} {len(pairs)}"]
    for (b1, b2) in pairs:
        out.append(f"{b1} {b2}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
