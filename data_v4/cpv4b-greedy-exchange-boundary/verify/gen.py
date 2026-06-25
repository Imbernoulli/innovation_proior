import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 12)            # keep small so brute's 2^n stays feasible
    coord = random.randint(1, 12)        # tiny coordinate range -> many touching/overlapping ends

    lines = [str(n)]
    for _ in range(n):
        a = random.randint(1, coord)
        b = random.randint(1, coord)
        l, r = min(a, b), max(a, b)
        lines.append(f"{l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
