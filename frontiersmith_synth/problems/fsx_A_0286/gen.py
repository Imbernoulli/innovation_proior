import sys, random

# fsx_A_0286  -- integer-matrix-construction, variant #7, "warehouse robotics"
# Difficulty ladder: an N x N sign-polarity array with a set of PRE-WIRED (fixed)
# cells that the solver must respect. N is ODD (9,11,...,23) so NO Hadamard matrix
# exists at these orders -- the max-|det| optimum is genuinely unknown, and the
# fixed cells forbid dropping in any known closed-form construction.

def main():
    testId = int(sys.argv[1])
    n = 7 + 2 * testId                 # 9, 11, 13, ..., 23  (all odd)
    rng = random.Random(90000 + testId * 7919)
    nf = int(round(0.15 * n * n))      # ~15% of cells are pre-wired
    cells = list(range(n * n))
    rng.shuffle(cells)
    cells = cells[:nf]
    fixed = []
    for c in cells:
        fixed.append((c // n, c % n, rng.choice((-1, 1))))
    fixed.sort()

    out = ["%d %d" % (n, len(fixed))]
    for (r, c, v) in fixed:
        out.append("%d %d %d" % (r, c, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
