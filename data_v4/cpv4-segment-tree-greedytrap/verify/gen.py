import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 8)
    q = rng.randint(1, 10)
    # values include negatives, zeros, positives so contiguity actually bites
    lo, hi = -9, 9
    a = [rng.randint(lo, hi) for _ in range(n)]

    lines = []
    lines.append(f"{n} {q}")
    lines.append(" ".join(map(str, a)))
    for _ in range(q):
        t = rng.randint(1, 2)
        if t == 1:
            p = rng.randint(1, n)
            v = rng.randint(lo, hi)
            lines.append(f"1 {p} {v}")
        else:
            l = rng.randint(1, n)
            r = rng.randint(l, n)
            lines.append(f"2 {l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
