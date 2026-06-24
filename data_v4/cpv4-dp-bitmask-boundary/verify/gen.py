import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so brute force (subset enumeration) stays fast.
    m = rng.randint(1, 8)          # number of slots
    n = rng.randint(0, 7)          # number of bands

    lines = [f"{m} {n}"]
    for _ in range(n):
        # Deliberately generate some bands that fit and some that run off the end,
        # to exercise the inclusive/exclusive fit boundary (s + d <= m).
        s = rng.randint(0, m)              # may equal m (then it cannot fit)
        d = rng.randint(1, m)              # length 1..m
        p = rng.randint(1, 20)            # strictly positive profit
        lines.append(f"{s} {d} {p}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
