import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 9)
    if n == 0:
        L = rng.randint(1, 3)
        print(f"{n} {L}")
        return
    L = rng.randint(1, n)
    # Mix of small negatives and positives to exercise the greedy trap and the
    # "leave a hole" decisions; occasional zeros.
    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            vals.append(rng.randint(-9, -1))
        elif r < 0.55:
            vals.append(0)
        else:
            vals.append(rng.randint(1, 9))
    print(f"{n} {L}")
    print(" ".join(map(str, vals)))

if __name__ == "__main__":
    main()
