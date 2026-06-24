import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 12)
    # small coordinate range so overlaps and exact touching (f == s) happen often
    T = rng.choice([4, 6, 8, 10])
    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, T)
        b = rng.randint(0, T)
        if a == b:
            b = a + 1  # ensure positive length (s < f)
        s, f = min(a, b), max(a, b)
        # profits: include some that make "earliest finish" greedy tempting but wrong
        p = rng.randint(1, 20)
        lines.append(f"{s} {f} {p}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
