import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # tiny cases so the brute-force bitmask cover stays cheap
    n = rng.randint(0, 8)
    L = rng.randint(1, 6)
    # small coordinate range so boundary collisions (t == s+L) happen often
    coord_max = rng.randint(1, 12)
    t = [rng.randint(0, coord_max) for _ in range(n)]
    out = [f"{n} {L}"]
    out.append(" ".join(map(str, t)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
