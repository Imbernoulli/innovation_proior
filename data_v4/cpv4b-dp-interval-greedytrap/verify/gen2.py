import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 14)
    T = rng.randint(3, 20)
    lines = [str(n)]
    for _ in range(n):
        s = rng.randint(0, T - 1)
        e = rng.randint(s + 1, T)
        v = rng.randint(0, 100)
        lines.append(f"{s} {e} {v}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
