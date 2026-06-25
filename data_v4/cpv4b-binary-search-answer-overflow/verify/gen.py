import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the linear-scan brute force terminates quickly.
    n = rng.randint(1, 6)
    # periods small -> answer T stays small enough for brute force
    s = [rng.randint(1, 8) for _ in range(n)]
    # P chosen so the answer T is modest. With smallest period 1, T ~ P at most.
    P = rng.randint(1, 40)

    print(n, P)
    print(' '.join(map(str, s)))

main()
