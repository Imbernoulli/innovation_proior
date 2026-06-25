import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 6)
    m = rng.randint(1, 6)
    # Mix small values with occasional large (overflow-pressuring) values.
    def val():
        r = rng.random()
        if r < 0.4:
            return rng.randint(1, 5)
        elif r < 0.7:
            return rng.randint(1, 10**6)
        else:
            return rng.randint(1, 4 * 10**9)
    a = [val() for _ in range(n)]
    b = [val() for _ in range(m)]
    K = rng.randint(1, n * m)

    out = []
    out.append(f"{n} {m} {K}")
    out.append(" ".join(map(str, a)))
    out.append(" ".join(map(str, b)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
