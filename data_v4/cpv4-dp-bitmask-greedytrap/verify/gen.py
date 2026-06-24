import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Tiny cases so the brute force (2^k subset enumeration) is feasible.
    m = rng.randint(1, 6)          # up to 6 skills
    k = rng.randint(1, 12)         # up to 12 contractors -> 2^12 = 4096 subsets in brute
    FULL = (1 << m) - 1

    lines = [f"{m} {k}"]
    for _ in range(k):
        cost = rng.randint(1, 20)
        # Bias toward small skill sets so greedy-vs-optimal divergence is common,
        # but allow the full range 0 .. FULL.
        r = rng.random()
        if r < 0.5:
            # 1 or 2 random skills
            bits = rng.randint(1, max(1, min(2, m)))
            s = 0
            for _b in range(bits):
                s |= (1 << rng.randint(0, m - 1))
        else:
            s = rng.randint(0, FULL)
        lines.append(f"{cost} {s}")

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
