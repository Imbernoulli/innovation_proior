import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Tiny cases so brute's millisecond scan terminates quickly and the
    # answer T stays small. We keep w, c, N, m all small here.
    m = rng.randint(1, 5)
    # keep N small so brute scan is cheap
    N = rng.randint(0, 30)
    lines = [f"{m} {N}"]
    for _ in range(m):
        w = rng.randint(0, 15)   # warm-up delay
        c = rng.randint(1, 10)   # cycle time (>=1)
        lines.append(f"{w} {c}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
