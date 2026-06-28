import sys
import random
from math import gcd

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep the lcm small so the brute force can scan [0, lcm).
    # Pool of small moduli, deliberately NON-COPRIME (shared factors 2,3) so the
    # gcd-divisibility merge is actually exercised.
    pool = [2, 3, 4, 6, 8, 9, 12, 5, 10, 15]

    k = rng.randint(1, 6)
    # Pick moduli, but cap the running lcm so the brute stays fast.
    mods = []
    L = 1
    attempts = 0
    while len(mods) < k and attempts < 200:
        attempts += 1
        m = rng.choice(pool)
        nl = L // gcd(L, m) * m
        if nl <= 5000:
            mods.append(m)
            L = nl
    if not mods:
        mods = [rng.choice(pool)]
        L = mods[0]
    k = len(mods)

    # Hidden true x => a guaranteed-consistent base.
    x = rng.randrange(0, max(L, 1))
    rems = [x % m for m in mods]

    # With some probability, corrupt one remainder (often => contradiction,
    # but sometimes still consistent — both are valid test cases).
    if rng.random() < 0.45 and k >= 2:
        j = rng.randrange(k)
        rems[j] = rng.randrange(0, mods[j])

    # Also randomly add an offset to remainders to test the % normalization path:
    # occasionally emit remainders outside [0, mi) including negatives? The contract
    # says 0 <= ri < mi, so keep them in range here (negative-ri is exercised by
    # explicit edge tests, not the random fuzz).

    lines = [str(k)]
    for r, m in zip(rems, mods):
        lines.append(f"{r} {m}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
