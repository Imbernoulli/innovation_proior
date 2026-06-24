import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small alphabet biases toward periodic structure so the tiling condition
    # is exercised hard (many borders, many divisor near-misses).
    alpha_size = rng.choice([1, 2, 2, 2, 3])
    alphabet = "abcdefghij"[:alpha_size]

    n = rng.randint(1, 14)

    mode = rng.randint(0, 3)
    if mode == 0:
        # fully random
        s = "".join(rng.choice(alphabet) for _ in range(n))
    elif mode == 1:
        # a repeated tile (guarantees tiled prefixes / divisor boundaries)
        d = rng.randint(1, max(1, n // 2))
        tile = "".join(rng.choice(alphabet) for _ in range(d))
        reps = (n + d - 1) // d
        s = (tile * reps)[:n]
    elif mode == 2:
        # tile then a small perturbation tail (near-miss on divisibility)
        d = rng.randint(1, max(1, n // 2))
        tile = "".join(rng.choice(alphabet) for _ in range(d))
        reps = (n + d - 1) // d
        s = list((tile * reps)[:n])
        if s:
            j = rng.randrange(len(s))
            s[j] = rng.choice(alphabet)
        s = "".join(s)
    else:
        # almost-all-same with a couple of differing chars
        c = rng.choice(alphabet)
        s = list(c * n)
        for _ in range(rng.randint(0, 2)):
            if s:
                s[rng.randrange(len(s))] = rng.choice(alphabet)
        s = "".join(s)

    print(s)

main()
