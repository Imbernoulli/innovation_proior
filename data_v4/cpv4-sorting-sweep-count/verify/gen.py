import sys, random

seed = int(sys.argv[1])
random.seed(seed)

# Small cases so the O(n^2) brute is fast. Vary L and D to hit the wrap-around
# and the 2D >= L "everything counts" regimes.
n = random.randint(0, 12)
L = random.randint(1, 20)
# D ranges from 0 up to L (D >= L would make every pair count trivially; we still
# allow D close to L to stress the 2D >= L edge case).
D = random.randint(0, L)
p = [random.randint(0, L - 1) for _ in range(n)]

out = []
out.append(f"{n} {L} {D}")
if n > 0:
    out.append(" ".join(map(str, p)))
else:
    out.append("")
print("\n".join(out))
