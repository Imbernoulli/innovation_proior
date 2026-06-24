import sys, random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)
    n = rng.randint(0, 30)
    T = rng.randint(1, 20)
    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, T); b = rng.randint(0, T)
        s, e = min(a, b), max(a, b)
        if rng.random() < 0.1:
            e = s
        w = rng.randint(1, 1000)
        lines.append(f"{s} {e} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
