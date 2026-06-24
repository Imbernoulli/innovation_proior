import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 20)
    coord = rng.choice([3, 5, 8, 12, 30])
    mode = rng.choice(["uniform", "points", "nested", "wide"])
    iv = []
    for _ in range(n):
        if mode == "points":
            a = rng.randint(0, coord); b = a
        elif mode == "wide":
            a = rng.randint(0, coord); b = rng.randint(a, coord)
        elif mode == "nested":
            a = rng.randint(0, coord); b = rng.randint(a, min(coord, a + rng.randint(0, 4)))
        else:
            a = rng.randint(0, coord); b = rng.randint(0, coord)
            if a > b: a, b = b, a
        iv.append((a, b))
    print(n)
    for (l, r) in iv:
        print(l, r)

main()
