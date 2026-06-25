import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the subset-enumeration brute force is feasible (n <= 14).
    n = rng.randint(0, 12)
    # Keep weights small relative to W so that an (incorrect) unbounded/reuse baseline
    # would actually differ -- this stresses the 0/1 constraint.
    W = rng.randint(0, 20)
    lines = [f"{n} {W}"]
    for _ in range(n):
        e = rng.randint(1, 8)           # positive energy cost
        v = rng.randint(0, 15)          # non-negative science value
        lines.append(f"{e} {v}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
