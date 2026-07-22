# TIER: trivial
# Reproduces the checker's internal baseline: a fixed-seed random size-k residue set.
import sys, random

def main():
    tk = sys.stdin.buffer.read().split()
    it = iter(tk)
    t = int(next(it)); ms = [int(next(it)) for _ in range(t)]; k = int(next(it))
    n = 1
    for m in ms:
        n *= m
    rng = random.Random(987654321)
    B = rng.sample(range(n), k)
    print(k)
    sys.stdout.write("\n".join(map(str, B)) + "\n")

if __name__ == "__main__":
    main()
