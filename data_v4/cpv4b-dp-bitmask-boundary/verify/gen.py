import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    m = rng.randint(1, 6)            # small number of vents so brute (2^n) stays cheap
    n = rng.randint(1, 8)            # small number of strips for the 2^n brute force

    lines = [f"{m} {n}"]
    for _ in range(n):
        a = rng.randint(0, m - 1)
        # b is the EXCLUSIVE upper bound: a < b <= m, so the span [a,b) is non-empty.
        b = rng.randint(a + 1, m)
        c = rng.randint(0, 20)       # costs may be zero
        lines.append(f"{a} {b} {c}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
