import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Small cases so the O(n^2) brute force is fast, but include n=0 and n=1.
    n = rng.randint(0, 40)
    # Mix of value ranges, including 0 and larger values to stress accumulation.
    hi = rng.choice([0, 1, 5, 100, 10000])
    a = [rng.randint(0, hi) for _ in range(n)]
    out = [str(n)]
    out.append(' '.join(map(str, a)))
    print('\n'.join(out))

main()
