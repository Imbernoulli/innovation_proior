import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Keep n small enough for the exponential-flavoured brute to finish quickly.
    n = rng.randint(1, 7)
    # Mix small bit-widths so XOR/OR interactions are rich but values collide.
    width = rng.choice([2, 3, 4, 5, 6])
    hi = (1 << width) - 1
    vals = [rng.randint(0, hi) for _ in range(n)]
    print(n)
    print(" ".join(map(str, vals)))

main()
