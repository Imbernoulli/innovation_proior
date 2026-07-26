import sys, random


def build(test_id):
    rng = random.Random(20000 + 97 * test_id)

    Q = 12 + 2 * test_id                      # inflow rate, grows with difficulty
    Dmax = 40
    Lmax = 6000
    M = 10                                    # max basins in the cascade
    PenNum, PenDen = 3, 2                     # contamination penalty multiplier P = 3/2

    # 7 distinct settling-time thresholds (residence time a class needs to settle),
    # drawn from a range that widens with test_id (difficulty ladder).
    pool_hi = 4 + 10 + 4 * test_id
    thr_ranks = sorted(rng.sample(range(3, 3 + pool_hi), 7))   # ascending: rank1 < rank2 < ... < rank7

    # permutation: which class id (1..7, the order they appear in the input) gets
    # which rank of threshold. This is the "radix sort in disguise" trap: an
    # id-ordered cascade only works if id order already equals threshold order.
    perm = list(range(1, 8))  # perm[r-1] = class id holding rank r (1-indexed rank)
    if test_id == 1:
        pass  # identity: id order == threshold order (no trap, warm-up case)
    elif test_id == 2:
        perm[0], perm[1] = perm[1], perm[0]  # mild single swap
    elif test_id == 3:
        perm[2], perm[3] = perm[3], perm[2]  # a different mild single swap
    else:
        # strong derangement trap: id order 1..7 is scrambled well away from
        # threshold-rank order, so an id-ordered cascade front-loads mismatched
        # classes into early basins (>=7 of the 10 cases are this kind of trap).
        while True:
            rng.shuffle(perm)
            if all(perm[k] != k + 1 for k in range(7)):
                break

    class_threshold = {}
    for r in range(7):
        class_threshold[perm[r]] = thr_ranks[r]

    class_mass = {c: rng.randint(70, 110) for c in range(1, 8)}
    class_value = {c: rng.randint(12, 18) for c in range(1, 8)}
    class_scour = {}
    for c in range(1, 8):
        req_depth = rng.randint(2, 5)
        class_scour[c] = -(-Q // req_depth)   # ceil(Q / req_depth)

    vol_needed = sum(Q * class_threshold[c] for c in range(1, 8))
    VolumeBudget = int(1.3 * vol_needed) + 7 * Dmax

    lines = []
    lines.append(f"{Q} {VolumeBudget} {M} {Dmax} {Lmax}")
    lines.append(f"{PenNum} {PenDen}")
    for c in range(1, 8):
        lines.append(f"{class_threshold[c]} {class_mass[c]} {class_value[c]} {class_scour[c]}")
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(build(test_id))


if __name__ == "__main__":
    main()
