import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Tiny cases so the O(n^2) brute is fast and exhaustive.
    n = random.randint(0, 12)
    m = random.randint(1, 8)
    t = random.randint(0, m - 1)        # documented: 0 <= t < m
    vals = [random.randint(0, 30) for _ in range(n)]

    out = []
    out.append(f"{n} {m} {t}")
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
