import sys, random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Small random cases (n small so the O(n^2) brute force stays fast).
# Value regimes by seed; the large regime uses the contract's cap (<= 30000) so that
# products (up to ~9e8) and their accumulation exercise the 64-bit requirement.
n = random.randint(0, 12)
regime = seed % 4
if regime == 0:
    hi = 10            # many ties / inversions, tiny values
elif regime == 1:
    hi = 5
elif regime == 2:
    hi = 1000
else:
    hi = 30000         # contract maximum: products ~9e8, sums overflow 32-bit fast

vals = [str(random.randint(1, hi)) for _ in range(n)]
print(n)
if n > 0:
    print(" ".join(vals))
