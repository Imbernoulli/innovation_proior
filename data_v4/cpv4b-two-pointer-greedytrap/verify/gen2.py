import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 14)
    # mix of tiny and big weights -> the greedy-trap structure (cheap edges that
    # waste budget vs. fewer-but-better picks from the other end).
    w = []
    for _ in range(n):
        if rng.random() < 0.45:
            w.append(rng.randint(1, 2))
        else:
            w.append(rng.randint(1, 40))
    total = sum(w)
    hi = max(total, 1)
    B = rng.randint(0, hi + 3)

    out = [f"{n} {B}", " ".join(map(str, w))]
    sys.stdout.write("\n".join(out) + "\n")

main()
