import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 12)

    # Mix of regimes so cross-product overflow is genuinely exercised:
    #  - "big": cal near 10^13 (forces __int128 in comparisons)
    #  - "small": tiny cal/mass to test ratio ties and reductions
    regime = rng.choice(["big", "small", "mixed", "tie"])

    cal = []
    mass = []
    for i in range(n):
        if regime == "big":
            c = rng.randint(10**12, 10**13)
            m = rng.randint(1, 1000)
        elif regime == "small":
            c = rng.randint(1, 50)
            m = rng.randint(1, 20)
        elif regime == "tie":
            # many items share the same density to stress fraction comparison
            base = rng.choice([2, 3, 5, 7])
            m = rng.randint(1, 30)
            c = base * m
            # occasionally perturb to break a tie slightly
            if rng.random() < 0.3:
                c += rng.randint(-1, 1)
                if c < 1:
                    c = 1
        else:  # mixed
            c = rng.randint(1, 10**13)
            m = rng.randint(1, 1000)
        cal.append(c)
        mass.append(m)

    sumMass = sum(mass)
    L = rng.randint(1, sumMass)

    out = [f"{n} {L}"]
    for i in range(n):
        out.append(f"{cal[i]} {mass[i]}")
    print("\n".join(out))

main()
