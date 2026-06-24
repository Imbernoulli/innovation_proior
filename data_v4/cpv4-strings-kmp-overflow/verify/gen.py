import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small alphabet so prefixes recur often (exercises the border chains).
    n = rng.randint(1, 14)
    alpha_size = rng.choice([1, 2, 2, 3])  # bias toward tiny alphabets
    alphabet = "abcdefghij"[:alpha_size]
    s = "".join(rng.choice(alphabet) for _ in range(n))

    sys.stdout.write(s + "\n")

main()
