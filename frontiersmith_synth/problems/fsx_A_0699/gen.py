import sys

# Rose window lead-line design: polar grid disk (R rings x A sectors), rotation
# order k, palette size p, harmony wheel over colors.
# Difficulty ladder deterministic in testId only. >=5 of the 10 cases use an ODD
# rotation order k (the trap: the "obvious" k-wedge two-colour design is then an
# odd cycle and has NO valid 2-colouring, regardless of p).
#
# A is always built as A = 2*k*w so that BOTH the naive k-wedge resolution and the
# doubled 2k-wedge resolution divide A evenly (integer sector widths).
LADDER = [
    # (R,   k,  w,  p)
    (4,   2,  3,  2),   # 1  small, k even
    (4,   3,  2,  2),   # 2  small, k ODD trap, p=2
    (6,   4,  2,  3),   # 3  small, k even
    (6,   4,  3,  3),   # 4  medium, k even
    (9,   6,  3,  4),   # 5  medium, k even
    (8,   7,  2,  2),   # 6  medium, k ODD trap, p=2
    (12,  8,  2,  5),   # 7  medium, k even
    (12,  8,  3,  3),   # 8  large, k even
    (16, 10,  3,  6),   # 9  large, k even
    (18, 11,  2,  2),   # 10 large adversarial, k ODD trap, p=2
]

# Deterministic LCG per testId, used only to fill the harmony wheel + weights.
def rng(seed):
    state = [seed & 0xFFFFFFFF]
    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0]
    return nxt


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    R, k, w, p = LADDER[idx]
    A = 2 * k * w

    nxt = rng(1000003 * i + 17)

    def frnd(lo, hi):
        return lo + (hi - lo) * (nxt() % 1000003) / 1000003.0

    we = round(frnd(1.0, 4.0), 4)
    wh = round(frnd(1.0, 4.0), 4)

    # symmetric harmony wheel, positive entries, zero diagonal
    H = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(a + 1, p):
            v = round(frnd(0.3, 1.0), 4)
            H[a][b] = v
            H[b][a] = v

    print(R, A, k, p)
    print("%.4f %.4f" % (we, wh))
    for row in H:
        print(" ".join("%.4f" % x for x in row))


if __name__ == "__main__":
    main()
