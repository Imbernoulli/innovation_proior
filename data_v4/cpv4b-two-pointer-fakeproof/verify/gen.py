import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases to stress the two-pointer / bit-count maintenance.
    n = rng.randint(0, 8)
    # Keep values small so windows shrink/grow often and OR overlaps K a lot.
    maxv = rng.choice([1, 3, 7, 15, 31, 63])
    a = [rng.randint(0, maxv) for _ in range(n)]
    # Choose K in a range that makes the boundary interesting.
    hi = (max(a) | (maxv)) if a else maxv
    K = rng.randint(0, hi + 2)

    out = [f"{n} {K}"]
    if n > 0:
        out.append(" ".join(map(str, a)))
    else:
        out.append("")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
