import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep n small so brute force over n! permutations is feasible.
    n = rng.randint(0, 7)
    # Occasionally force tiny edge cases.
    r = rng.random()
    if r < 0.10:
        n = 0
    elif r < 0.20:
        n = 1
    elif r < 0.30:
        n = 2

    # Mix of value ranges to exercise ties and asymmetry.
    hi = rng.choice([0, 1, 3, 5, 9, 20])

    out = [str(n)]
    for i in range(n):
        row = [str(rng.randint(0, hi)) for _ in range(n)]
        out.append(" ".join(row))
    for i in range(n):
        row = [str(rng.randint(0, hi)) for _ in range(n)]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
