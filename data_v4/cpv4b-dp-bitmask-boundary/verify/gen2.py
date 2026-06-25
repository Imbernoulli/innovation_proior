import sys, random

# Variant generator that can leave some vents uncoverable (to exercise the -1 path):
# it restricts strips to the left half, so right-side vents may never be sealed.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    m = rng.randint(1, 6)
    n = rng.randint(1, 7)

    lines = [f"{m} {n}"]
    cap = max(1, m - rng.randint(0, 2))   # strips may be confined to [0, cap)
    for _ in range(n):
        a = rng.randint(0, max(0, cap - 1))
        b = rng.randint(a + 1, cap)
        c = rng.randint(0, 15)
        lines.append(f"{a} {b} {c}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
