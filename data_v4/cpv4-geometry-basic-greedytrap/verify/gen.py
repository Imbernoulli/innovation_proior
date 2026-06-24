import random, sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 8)
    L = rng.randint(0, 12)
    coord = rng.randint(0, 20)
    xs = [rng.randint(-coord, coord) for _ in range(n)]
    out = [f"{n} {L}"]
    out.append(" ".join(map(str, xs)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
