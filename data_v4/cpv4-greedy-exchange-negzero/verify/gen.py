import sys
import random

# Small-case generator parameterized by an integer seed.
# Deliberately stresses the sign/base-case pitfall: deadlines may be 0, payouts
# may be negative or zero, n may be 0, and sometimes the whole instance is
# all-nonpositive (so the answer must be 0).

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Bias toward tiny n (brute enumerates 2^n), include n = 0 occasionally.
    r = rng.random()
    if r < 0.08:
        n = 0
    else:
        n = rng.randint(1, 9)

    # Choose an instance "flavor" to cover corners.
    flavor = rng.randint(0, 4)

    d = []
    v = []
    for _ in range(n):
        # deadlines: allow 0 (invalid day) up to a bit more than n
        d.append(rng.randint(0, n + 2))
        if flavor == 0:
            # all nonpositive payouts -> answer must be 0
            v.append(rng.randint(-9, 0))
        elif flavor == 1:
            # all zero payouts -> answer must be 0
            v.append(0)
        elif flavor == 2:
            # mix of negatives, zeros, positives
            v.append(rng.randint(-9, 9))
        elif flavor == 3:
            # mostly positive, some heavy
            v.append(rng.choice([rng.randint(-3, 0), rng.randint(1, 20)]))
        else:
            # small magnitudes to force ties
            v.append(rng.randint(-2, 3))

    out = [str(n)]
    out.append(" ".join(map(str, d)))
    out.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
