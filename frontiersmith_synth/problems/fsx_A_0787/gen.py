import sys, random

# testId -> (K voices, delay list, L_cap, density of off-diagonal consonance)
# density is deliberately LOW on several cases (trap cases): a sparse consonance
# table makes a single-pass, no-lookahead melody writer paint itself into a
# corner (dead end), while a search that treats the problem as a walk in the
# product automaton of shifted copies can route around the trap.
PLAN = {
    1: (1, [3],           10, 0.55),
    2: (1, [4],           12, 0.50),
    3: (2, [2, 5],        18, 0.32),  # trap
    4: (2, [3, 4],        20, 0.45),
    5: (2, [2, 7],        24, 0.28),  # trap
    6: (3, [2, 3, 5],     28, 0.40),
    7: (3, [3, 5, 8],     32, 0.26),  # trap
    8: (2, [4, 9],        34, 0.35),
    9: (3, [2, 6, 9],     38, 0.24),  # trap
    10: (3, [3, 7, 10],   40, 0.24),  # trap, largest / adversarial
}


def main():
    tid = int(sys.argv[1])
    K, delays, L, density = PLAN[tid]
    rng = random.Random(1000003 * tid + 7)

    # Consonance matrix: symmetric 12x12, 0/1. Diagonal always 1 (a note is
    # always consonant with an exact unison of itself). Off-diagonal entries
    # are i.i.d. Bernoulli(density), mirrored for symmetry.
    C = [[0] * 12 for _ in range(12)]
    for i in range(12):
        C[i][i] = 1
    for i in range(12):
        for j in range(i + 1, 12):
            v = 1 if rng.random() < density else 0
            C[i][j] = v
            C[j][i] = v

    # transpositions: random, but avoid all-zero (which would make the
    # problem degenerate / too close to a plain round with no transposition).
    voices = []
    for d in delays:
        t = rng.randrange(1, 12)
        voices.append((d, t))
    rng.shuffle(voices)

    out = []
    out.append(f"{K} {L}")
    for row in C:
        out.append(" ".join(map(str, row)))
    for d, t in voices:
        out.append(f"{d} {t}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
