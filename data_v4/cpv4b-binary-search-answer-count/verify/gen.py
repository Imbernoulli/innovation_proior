import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the brute force (walking minutes) terminates fast.
    m = rng.randint(1, 4)
    # small primes/composites; allow duplicates to exercise dedup of equal p_i
    pool = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    p = [rng.choice(pool) for _ in range(m)]
    # k small enough that the k-th lit minute is tiny
    k = rng.randint(1, 40)

    out = [f"{m} {k}"]
    out.append(" ".join(map(str, p)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
