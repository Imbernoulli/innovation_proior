import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small-case sizes for differential testing.
    n = rng.randint(1, 12)
    # Bias alphabet small so substrings collide often (stresses LCP / equality paths).
    alpha = rng.choice([1, 2, 2, 3, 4])
    s = "".join(chr(ord('a') + rng.randrange(alpha)) for _ in range(n))

    q = rng.randint(1, 15)
    lines = [s, str(q)]
    for _ in range(q):
        l1 = rng.randint(1, n)
        len1 = rng.randint(1, n - l1 + 1)
        l2 = rng.randint(1, n)
        len2 = rng.randint(1, n - l2 + 1)
        lines.append(f"{l1} {len1} {l2} {len2}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
