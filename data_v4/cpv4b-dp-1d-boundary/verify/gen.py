import sys, random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

n = random.randint(1, 12)
D = random.randint(1, max(1, n))  # small D so the reach boundary bites
vals = []
for i in range(n):
    r = random.random()
    if i != 0 and r < 0.25:
        vals.append(-1)            # broken stone
    else:
        vals.append(random.randint(0, 20))

print(n, D)
print(" ".join(map(str, vals)))
