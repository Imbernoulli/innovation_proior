import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes so the window genuinely slides, sometimes empty,
    # sometimes covering the whole array.
    n = rng.randint(0, 12)
    maxw = rng.choice([0, 1, 3, 7, 20])
    w = [rng.randint(0, maxw) for _ in range(n)]
    total = sum(w)
    # Choose S across the interesting range: below smallest, mid, above total.
    choices = [0, 1]
    if total > 0:
        choices += [rng.randint(0, total), total, total + rng.randint(0, 5)]
    choices += [rng.randint(0, max(1, maxw))]
    S = rng.choice(choices)

    out = [f"{n} {S}"]
    out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
