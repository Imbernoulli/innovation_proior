import sys, random
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed*7+3)
n = random.randint(1, 25)
D = random.randint(1, max(1, random.choice([1, 1, 2, 3, n, n+2])))
vals = []
for i in range(n):
    if i != 0 and random.random() < 0.35:
        vals.append(-1)
    else:
        vals.append(random.randint(0, 30))
print(n, D)
print(" ".join(map(str, vals)))
