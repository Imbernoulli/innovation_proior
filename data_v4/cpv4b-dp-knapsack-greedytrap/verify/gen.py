import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Small cases so the bitmask brute force is feasible (n <= 16).
    n = random.randint(0, 14)
    C = random.randint(0, 25)
    lines = ["{} {}".format(n, C)]
    for _ in range(n):
        # Allow some crates heavier than C, and zero-weight / zero-value crates,
        # to probe corners. Weights and values are non-negative.
        w = random.randint(0, 30)
        v = random.randint(0, 30)
        lines.append("{} {}".format(w, v))
    sys.stdout.write("\n".join(lines) + "\n")

main()
