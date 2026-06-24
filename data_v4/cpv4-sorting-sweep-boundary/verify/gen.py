import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases, with a SMALL coordinate range so that endpoints collide often
    # (this is exactly where the inclusive/exclusive boundary bug shows up).
    n = rng.randint(0, 8)
    coord_max = rng.choice([2, 3, 4, 5, 6])

    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, coord_max)
        b = rng.randint(0, coord_max)
        # Sometimes allow degenerate s >= e to exercise the guard.
        if rng.random() < 0.15:
            s, e = a, b               # may be degenerate
        else:
            s, e = min(a, b), max(a, b)
            if s == e:                # force a real interval most of the time
                e = s + 1
        lines.append(f"{s} {e}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
