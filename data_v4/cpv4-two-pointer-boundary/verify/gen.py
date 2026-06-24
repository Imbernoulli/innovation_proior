import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 9)
    # Small value range so windows frequently become valid/invalid, and small D
    # so the boundary (where the window just barely (in)validates) is exercised.
    vmax = random.choice([2, 3, 5, 10])
    D = random.choice([0, 1, 2, 3, 5])
    vals = [random.randint(-vmax, vmax) for _ in range(n)]

    out = []
    out.append(f"{n} {D}")
    out.append(" ".join(str(v) for v in vals))
    sys.stdout.write("\n".join(out) + "\n")

main()
