import sys, random

# Difficulty ladder: frame size n grows with testId.
# NONE of these are multiples of 4, so a fully orthogonal (Hadamard) layout is
# impossible and the minimum interference energy is bounded strictly above zero
# (no degenerate E=0 optimum that would let the score saturate/divide-by-zero).
LADDER = [6, 7, 9, 10, 11, 13, 14, 15, 17, 18]

def main():
    tid = int(sys.argv[1])
    idx = max(1, min(len(LADDER), tid)) - 1
    n = LADDER[idx]
    rng = random.Random(70000 + tid)
    r0 = [rng.choice((-1, 1)) for _ in range(n)]
    out = [str(n), " ".join(str(x) for x in r0)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
