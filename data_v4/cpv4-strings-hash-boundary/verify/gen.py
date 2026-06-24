import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small alphabet makes nontrivial periods (and tricky boundaries) common.
    alpha_size = rng.choice([1, 2, 2, 2, 3, 3, 4])
    alphabet = "abcdefghij"[:alpha_size]

    n = rng.randint(1, 12)

    # Sometimes build a string with strong periodic structure to stress periods.
    mode = rng.randint(0, 2)
    if mode == 0:
        s = "".join(rng.choice(alphabet) for _ in range(n))
    elif mode == 1:
        # repeat a small block
        blen = rng.randint(1, max(1, n // 2 if n >= 2 else 1))
        block = "".join(rng.choice(alphabet) for _ in range(blen))
        s = (block * (n // blen + 1))[:n]
    else:
        # block + partial tail to create near-periodic substrings
        blen = rng.randint(1, max(1, n))
        block = "".join(rng.choice(alphabet) for _ in range(blen))
        reps = rng.randint(1, 4)
        s = (block * reps)
        s = s[:n] if len(s) >= n else s + "".join(rng.choice(alphabet) for _ in range(n - len(s)))
        n = len(s)

    q = rng.randint(1, 8)
    queries = []
    for _ in range(q):
        l = rng.randint(1, n)
        r = rng.randint(l, n)
        queries.append((l, r))

    lines = [s, str(q)]
    for (l, r) in queries:
        lines.append(f"{l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
