import sys, random

# fsx_G_0479 -- Interference-free frequency-comb (Sidon / B2 set) planning.
# `python3 gen.py <testId>` prints ONE instance:
#   line 1:  n k        (n = number of channels in the band; k = #forbidden channels)
#   line 2:  k forbidden channel indices (sorted, space separated; may be blank if k==0)
# Difficulty ladder: n grows with testId. Randomness seeded ONLY by testId (deterministic).
# Forbidden channels model already-occupied spectrum. They NEVER include a power of two,
# so the checker's power-of-two guard-band baseline is always fully available.

def gen(t):
    n = 300 * t * t
    if n < 100:
        n = 100
    if n > 12000:
        n = 12000
    rnd = random.Random(90000 + t)
    powers = set()
    v = 1
    while v <= n:
        powers.add(v)
        v *= 2
    candidates = [c for c in range(1, n + 1) if c not in powers]
    k = int(0.10 * n)
    if k > len(candidates):
        k = len(candidates)
    forbidden = sorted(rnd.sample(candidates, k)) if k > 0 else []
    return n, forbidden

if __name__ == "__main__":
    t = int(sys.argv[1])
    n, forb = gen(t)
    print("%d %d" % (n, len(forb)))
    print(" ".join(map(str, forb)))
