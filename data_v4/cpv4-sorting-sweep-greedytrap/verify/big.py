import random
rng = random.Random(7)
n = 200000
print(n)
out = []
C = 10**9
for _ in range(n):
    a = rng.randint(0, C); b = rng.randint(0, C)
    if a == b: b = a + 1
    s, f = min(a, b), max(a, b)
    p = rng.randint(1, 10**9)
    out.append(f"{s} {f} {p}")
print("\n".join(out))
