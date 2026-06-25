import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 9)
    # Values can be negative, zero, or positive so the negative-aware variant is exercised.
    lo, hi = rng.choice([(-5, 5), (-3, 8), (-9, 9), (0, 6), (-6, 0)])
    a = [rng.randint(lo, hi) for _ in range(n)]
    # S chosen near the achievable sum range so both "found" and "-1" outcomes appear.
    S = rng.randint(-8, 12)

    out = [f"{n} {S}"]
    out.append(" ".join(map(str, a)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
