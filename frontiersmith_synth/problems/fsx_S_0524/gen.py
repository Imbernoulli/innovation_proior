import sys

# Difficulty ladder. N is ODD (plain-weave stays fully symmetric; no perfect tiling).
# Trap cases (large N, small L, k=8) are 6..10: the monotone diamond draft cannot
# reach the motif diversity a factor-space de-Bruijn arrangement achieves.
# Small shaft/treadle budgets make S^k a HARD ceiling on the number of distinct k-windows
# (far below the (N-k+1)^2 window positions), so maximal motif diversity is a genuine
# de-Bruijn-style packing problem under the palindrome + float constraints -- not trivially
# saturable. N is ODD (plain weave stays fully symmetric; no perfect tiling).
#            N    S  T  L  k   ch cv cr
TABLE = {
    1:  (99,  3, 3, 5, 4,  1, 1, 1),
    2:  (111, 3, 3, 5, 4,  2, 1, 1),
    3:  (121, 3, 3, 4, 4,  1, 2, 1),
    4:  (121, 3, 3, 5, 4,  1, 1, 2),
    5:  (133, 3, 3, 5, 4,  1, 1, 1),
    6:  (133, 4, 4, 5, 4,  2, 2, 1),
    7:  (145, 3, 3, 5, 4,  1, 2, 2),
    8:  (145, 3, 3, 4, 4,  2, 1, 2),
    9:  (155, 3, 3, 5, 4,  1, 1, 1),
    10: (161, 3, 3, 5, 4,  1, 1, 1),
}

LAM_NUM = 1
LAM_DEN = 17   # diversity-floor mix lambda = 1/17 (keeps the score ceiling open)

def main():
    i = int(sys.argv[1])
    N, S, T, L, k, ch, cv, cr = TABLE[i]
    print("%d %d %d %d %d" % (N, S, T, L, k))
    print("%d %d %d" % (ch, cv, cr))
    print("%d %d" % (LAM_NUM, LAM_DEN))

if __name__ == "__main__":
    main()
