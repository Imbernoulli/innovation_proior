import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # n small enough that brute's 2^n enumeration is cheap.
    n = rng.randint(0, 14)

    # Small alphabet on purpose: heavy duplication is exactly where double-counting bites.
    # Mix in a few "wide" cases with a larger alphabet so unique-element paths are tested too.
    mode = rng.randint(0, 3)
    if mode == 0:
        alphabet = [1]                       # all identical
    elif mode == 1:
        alphabet = [1, 2]                    # binary
    elif mode == 2:
        alphabet = list(range(1, 4))         # 3 symbols
    else:
        alphabet = list(range(1, n + 2))     # mostly distinct (allow some collisions)

    t = [rng.choice(alphabet) for _ in range(n)]

    out = [str(n)] + [str(x) for x in t]
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
