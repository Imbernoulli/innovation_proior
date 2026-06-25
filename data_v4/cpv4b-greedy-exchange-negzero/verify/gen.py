import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the subset-enumeration brute force stays fast.
    n = rng.randint(0, 10)
    if n == 0:
        # Degenerate: no crates. Then L = K = 0 is the only valid pair.
        L = 0
        K = 0
    else:
        K = rng.randint(0, n)
        L = rng.randint(0, K)

    # Bias toward small magnitudes and lots of negatives/zeros to hammer the
    # sign / base-case logic; occasionally use larger magnitudes.
    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            vals.append(rng.randint(-6, 0))      # negatives and zeros
        elif r < 0.55:
            vals.append(0)                        # extra zeros
        elif r < 0.85:
            vals.append(rng.randint(-6, 6))       # mixed
        else:
            vals.append(rng.randint(-1000, 1000)) # occasional larger magnitude

    out = [f"{n} {L} {K}"]
    if n > 0:
        out.append(" ".join(str(v) for v in vals))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
