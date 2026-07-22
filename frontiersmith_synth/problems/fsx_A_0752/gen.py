import sys, random

# Difficulty ladder: (n, d, kind) grows small -> large/adversarial.
# d = max allowed run of identical bits inside any codeword (the channel constraint).
#
# kind="dominant": one clearly dominant symbol plus a tight, near-equal tail.
#   The plain (unconstrained) Huffman tree stays shallow and balanced enough to
#   already respect the run ceiling here, so it beats the flat baseline outright
#   -- these are the control cases that show the textbook recipe is not useless.
# kind="trap": steep geometric/power-law skew with a long thin tail. The plain
#   Huffman tree becomes a deep, comb-shaped branch (many bits in the same
#   direction in a row) that badly violates a small run ceiling -- these are the
#   planted trap cases.
LADDER = [
    (6,  4, "dominant"),  # 1 control, warm-up
    (8,  2, "trap"),      # 2 TRAP
    (10, 4, "dominant"),  # 3 control
    (10, 2, "trap"),      # 4 TRAP
    (15, 4, "dominant"),  # 5 control
    (15, 2, "trap"),      # 6 TRAP
    (20, 2, "trap"),      # 7 TRAP
    (25, 2, "trap"),      # 8 TRAP
    (30, 2, "trap"),      # 9 TRAP
    (40, 2, "trap"),      # 10 TRAP, largest
]


def gen_weights(rng, n, kind):
    if kind == "dominant":
        w = [rng.randint(9, 11) for _ in range(n - 1)]
        w.append(rng.randint(800, 1500))
        rng.shuffle(w)
    elif kind == "trap":
        w = []
        cur = rng.randint(3000, 5000)
        for i in range(n):
            w.append(max(1, cur))
            cur = max(1, int(cur * rng.uniform(0.22, 0.35)))
        rng.shuffle(w)
    else:
        w = [1] * n
    return w


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    n, d, kind = LADDER[idx]
    rng = random.Random(1000003 * i + 17)
    w = gen_weights(rng, n, kind)
    # keep weights bounded (fits comfortably in exact integer arithmetic)
    w = [min(x, 5000) for x in w]
    print(n, d)
    print(" ".join(str(x) for x in w))


if __name__ == "__main__":
    main()
