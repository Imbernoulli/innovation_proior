import sys, random

# Construct a full-constraint case engineered so the winning comparison
# happens between two large fractions P1/W1 vs P2/W2 where P*W' exceeds 2^63.
# n=50, cal up to 1e13, mass up to 1000 -> total value up to 5e14,
# total mass up to 5e4 -> cross product up to ~2.5e19 >> 9.2e18 (LL max).

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = 50
    items = []
    # Two near-tie high-density "anchor" clusters so the decisive comparison
    # is between big numerators; cross multiplication then overflows int64.
    for i in range(n):
        m = rng.randint(900, 1000)
        # density around 1e10 per mass unit -> cal ~ 1e13
        dens = rng.randint(9_000_000_000, 10_000_000_000)
        c = dens * m + rng.randint(-m, m)  # tiny perturbation breaks ties
        c = max(1, min(c, 10**13))
        items.append((c, m))
    sumMass = sum(m for _, m in items)
    L = rng.randint(sumMass // 2, sumMass)  # force large total weight -> big W
    out = [f"{n} {L}"]
    for c, m in items:
        out.append(f"{c} {m}")
    print("\n".join(out))

main()
