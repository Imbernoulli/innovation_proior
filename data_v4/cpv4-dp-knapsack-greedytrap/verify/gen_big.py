import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(0, 15)
    B = random.randint(0, 60)
    out = [f"{n} {B}"]
    for _ in range(n):
        c = random.randint(1, 40)   # costs strictly positive, per contract
        r = random.randint(0, 80)   # rewards non-negative, per contract
        out.append(f"{c} {r}")
    sys.stdout.write("\n".join(out) + "\n")

main()
