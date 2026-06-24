import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Tiny cases so the 2^n brute force is feasible, but exercise the structure:
# items that may or may not fit, zero values, ties, capacity occasionally < smallest weight.
n = random.randint(0, 10)
W = random.randint(0, 20)

lines = ["%d %d" % (n, W)]
for _ in range(n):
    w = random.randint(0, 12)
    # Occasionally use large values to stress accumulation (still safe for brute's Python ints).
    if random.random() < 0.3:
        v = random.randint(0, 1000000000)
    else:
        v = random.randint(0, 30)
    lines.append("%d %d" % (w, v))

sys.stdout.write("\n".join(lines) + "\n")
