import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # Bias toward small n (including 0 and 1) and toward negatives/zeros so the
    # all-negative / empty / sign-handling corners get hit frequently.
    r = random.random()
    if r < 0.10:
        n = 0
    elif r < 0.25:
        n = 1
    else:
        n = random.randint(2, 12)
    # Choose a value regime; some seeds are all-negative, some all-zero, some mixed.
    regime = random.randint(0, 4)
    out = [str(n)]
    vals = []
    for _ in range(n):
        if regime == 0:        # all negative
            v = random.randint(-9, -1)
        elif regime == 1:      # zeros and negatives
            v = random.randint(-5, 0)
        elif regime == 2:      # all zero
            v = 0
        elif regime == 3:      # full mixed including positives
            v = random.randint(-7, 7)
        else:                  # mostly positive with a few negatives
            v = random.randint(-2, 9)
        vals.append(str(v))
    out.append(" ".join(vals))
    print("\n".join(out))

main()
