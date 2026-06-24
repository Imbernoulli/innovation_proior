import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Small cases that exercise the inclusive/exclusive boundary heavily:
# - small n so single-element ranges [l, l] (which connect NOTHING) appear
# - ranges that touch at endpoints ([1,3] then [3,5]) and ranges separated by 1
n = random.randint(1, 8)
m = random.randint(0, 8)

lines = ["{} {}".format(n, m)]
for _ in range(m):
    l = random.randint(1, n)
    r = random.randint(l, n)   # l <= r, both in [1, n]
    lines.append("{} {}".format(l, r))

sys.stdout.write("\n".join(lines) + "\n")
