import sys, random


def main():
    testId = int(sys.argv[1])
    # difficulty ladder: grid grows, base varies, R always even (clean partner pairing)
    R_list = [4, 6, 6, 8, 8, 10, 10, 12, 12, 14]
    C_list = [7, 9, 11, 11, 13, 13, 15, 15, 17, 19]
    B_list = [8, 8, 9, 10, 9, 10, 9, 10, 10, 11]
    idx = min(max(testId, 1), 10) - 1
    R = R_list[idx]
    C = C_list[idx]
    base = B_list[idx]

    rng = random.Random(1000003 * testId + 7)

    cells = [(r, c) for r in range(R) for c in range(C)]
    rng.shuffle(cells)

    # ~55% of cells become counters, ~8% become seed hints, rest are free filler
    K = max(4, int(round(R * C * 0.55)))
    H = max(2, int(round(R * C * 0.08)))
    K = min(K, len(cells) - 2)
    H = min(H, len(cells) - K - 1)
    H = max(H, 0)

    counter_cells = cells[:K]
    hint_cells = cells[K:K + H]

    lines = []
    lines.append("%d %d %d" % (R, C, base))
    lines.append(str(K))
    for j, (r, c) in enumerate(counter_cells):
        d = rng.randrange(base)
        w = rng.randint(1, 9)
        # alternate ROW / PAIR so every instance mixes both scope types; PAIR is the
        # cross-row entangling mechanism that traps synchronous recount-and-rewrite
        scope = "ROW" if (j % 2 == 0) else "PAIR"
        lines.append("%d %d %d %d %s" % (r, c, d, w, scope))
    lines.append(str(len(hint_cells)))
    for (r, c) in hint_cells:
        v = rng.randrange(base)
        lines.append("%d %d %d" % (r, c, v))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
