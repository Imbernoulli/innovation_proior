import random, sys

seed = int(sys.argv[1])
rng = random.Random(seed)

# Bias toward small n and small values, including all-negative / empty / single corners,
# and threshold T that can be negative, zero, or positive.
mode = seed % 7
if mode == 0:
    n = 0
elif mode == 1:
    n = 1
elif mode == 2:
    n = rng.randint(2, 4)
else:
    n = rng.randint(0, 8)

# value range: sometimes all-negative, sometimes mixed, sometimes with zeros
vmode = rng.randint(0, 3)
if vmode == 0:
    lo, hi = -6, -1          # all negative
elif vmode == 1:
    lo, hi = -5, 5           # mixed with zeros
elif vmode == 2:
    lo, hi = 0, 0            # all zero
else:
    lo, hi = -8, 8

a = [rng.randint(lo, hi) for _ in range(n)]

# threshold can be negative, zero, positive, and around the sum extremes
T = rng.randint(-12, 12)

print(n, T)
print(' '.join(map(str, a)))
