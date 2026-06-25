import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 9)
    maxw = rng.choice([1, 3, 5, 8, 20])
    w = [rng.randint(1, maxw) for _ in range(n)]
    total = sum(w)
    hi = max(total, 1)
    # budget anywhere from "nothing fits" to "everything fits", with extra weight
    # on the interesting middle band where prefix and suffix compete.
    B = rng.randint(0, hi + 2)

    out = [f"{n} {B}", " ".join(map(str, w))]
    sys.stdout.write("\n".join(out) + "\n")

main()
