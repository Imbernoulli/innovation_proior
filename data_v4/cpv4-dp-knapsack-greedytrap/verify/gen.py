import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # tiny cases so brute force (2^n) is feasible
    n = random.randint(0, 12)
    B = random.randint(0, 40)

    lines = []
    lines.append(f"{n} {B}")
    for _ in range(n):
        # costs can exceed B sometimes (item never fits); rewards include 0 occasionally
        c = random.randint(1, 30)
        r = random.randint(0, 50)
        lines.append(f"{c} {r}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
