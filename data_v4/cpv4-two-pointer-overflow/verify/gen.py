import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases but exercising the structure. Keep values small enough that
    # brute force is fast, but allow B to range so windows of varying width
    # are selected (including the empty-window corner when B is tiny).
    n = rng.randint(0, 8)
    vmax = rng.choice([1, 3, 5, 10, 20])
    a = [rng.randint(1, vmax) for _ in range(n)]
    total = sum(a)
    # Choose B spanning [0, total + a little] so we hit: B too small for any
    # single element, B exactly equal to some window, B >= total.
    hi = total + rng.choice([0, 1, 3])
    B = rng.randint(0, hi if hi > 0 else 0)

    out = [f"{n} {B}"]
    out.append(" ".join(map(str, a)))
    print("\n".join(out))

main()
