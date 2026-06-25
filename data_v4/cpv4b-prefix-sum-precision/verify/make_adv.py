# Adversarial: large n, values that make cross products overflow int64.
# Layout: a long stretch of +1e9 then a final structure that forces the hull
# binary-search to compare two far-apart slopes whose cross product exceeds 9.2e18.
n = 200000
L = 1
# Make S grow then we want big slope differences. Use big positive then big negative.
# Put +1e9 for first half, -1e9 for second half. Prefix sums reach ~1e14.
vals = []
for i in range(n):
    if i < n//2:
        vals.append(10**9)
    else:
        vals.append(-10**9)
print(n, L)
print(' '.join(map(str, vals)))
