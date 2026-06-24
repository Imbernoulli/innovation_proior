import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    coord = rng.choice([6, 8, 10, 14])
    iv = []
    for _ in range(n):
        a = rng.randint(0, coord)
        b = rng.randint(0, coord)
        if a > b:
            a, b = b, a
        iv.append((a, b))

    print(n)
    for (l, r) in iv:
        print(l, r)

main()
