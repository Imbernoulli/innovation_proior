import sys, random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Mix of regimes so we hit: empty, single, all-negative, all-positive,
# zeros, and small mixed cases. The choice is seed-driven.
mode = seed % 7

if mode == 0:
    n = 0
elif mode == 1:
    n = 1
elif mode == 2:
    n = random.randint(1, 8)            # all negative
elif mode == 3:
    n = random.randint(1, 8)            # all positive
elif mode == 4:
    n = random.randint(0, 10)           # zeros heavy
else:
    n = random.randint(0, 10)           # fully mixed

vals = []
for _ in range(n):
    if mode == 2:
        vals.append(random.randint(-6, -1))
    elif mode == 3:
        vals.append(random.randint(1, 6))
    elif mode == 4:
        vals.append(random.choice([0, 0, 0, -2, 3, -1]))
    else:
        vals.append(random.randint(-6, 6))

out = [str(n)] + [str(v) for v in vals]
print(" ".join(out))
