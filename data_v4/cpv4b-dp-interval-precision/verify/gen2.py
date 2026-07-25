import sys, random

# larger / more adversarial small cases (still brute-able)
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 14)
    K = rng.randint(1, n)
    bmax = rng.choice([1, 4, 12, 40, 100])
    wmax = rng.choice([1, 4, 12, 40, 100])

    out = [f"{n} {K}"]
    for _ in range(n):
        b = rng.randint(-bmax, bmax)
        w = rng.randint(1, wmax)
        out.append(f"{b} {w}")
    # extreme ratios sometimes
    p = rng.choice([0, 1, rng.randint(0, 50), rng.randint(0, 1000)])
    q = rng.choice([1, rng.randint(1, 50), rng.randint(1, 1000)])
    out.append(f"{p} {q}")
    sys.stdout.write("\n".join(out) + "\n")

main()
