import sys

# testId -> (N, foldType).  N always even (no cell ever sits exactly ON the
# horizontal/vertical fold axes).  foldType in {2,4,8} sets the dihedral
# subgroup used to unfold the cuts (2 = one mirror, 4 = two mirrors, 8 = full
# square symmetry, i.e. the classic "fold-fold-fold-diagonally" snowflake cut).
CASES = [
    (16, 2), (18, 2),
    (20, 4), (22, 4), (24, 4),
    (26, 8), (28, 8), (30, 8), (34, 8), (40, 8),
]


def group_order(t):
    return t


def budget_for(N, t):
    target_area = N * N / 3.0
    g = group_order(t)
    v = int(round(1.20 * target_area / g))
    return max(8, v)


def main():
    tid = int(sys.argv[1])
    tid = max(1, min(len(CASES), tid))
    N, t = CASES[tid - 1]
    V = budget_for(N, t)
    print(N, t, V)


if __name__ == "__main__":
    main()
