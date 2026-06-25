import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Small cases so the 2^n brute force is feasible.
    n = random.randint(1, 12)
    C = random.randint(0, 30)
    lines = [f"{n} {C}"]
    for _ in range(n):
        w = random.randint(1, 12)
        # Mix small and large brightness values; large ones exercise the 64-bit path.
        if random.random() < 0.5:
            b = random.randint(1, 20)
        else:
            b = random.randint(900000000, 1000000000)
        lines.append(f"{w} {b}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
