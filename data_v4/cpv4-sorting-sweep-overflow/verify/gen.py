import sys, random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)

    n = rng.randint(0, 8)
    # Small coordinate range so simultaneous start/end ties are common -> exercises
    # the "ends before starts" tie-breaking in the half-open sweep.
    T = rng.randint(1, 6)
    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, T)
        b = rng.randint(0, T)
        s, e = min(a, b), max(a, b)
        # Occasionally force an empty interval (s == e) to test that path.
        if rng.random() < 0.15:
            e = s
        w = rng.randint(1, 9)
        lines.append(f"{s} {e} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
