import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 14)
    maxbits = rng.choice([2, 4, 8, 16, 30])
    maxv = (1 << maxbits) - 1
    a = [rng.randint(0, maxv) for _ in range(n)]
    hi = (max(a) if a else 0) | maxv
    K = rng.randint(0, hi + 3)
    out = [f"{n} {K}"]
    out.append(" ".join(map(str, a)) if n > 0 else "")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
