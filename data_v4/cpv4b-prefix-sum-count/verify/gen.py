import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress dedup / zero / negative / target-hitting.
    regime = rng.randint(0, 4)
    if regime == 0:
        n = rng.randint(0, 8)
        vlo, vhi = -3, 3
    elif regime == 1:
        n = rng.randint(0, 12)
        vlo, vhi = -2, 2       # lots of repeated prefix sums -> dedup pressure
    elif regime == 2:
        n = rng.randint(1, 10)
        vlo, vhi = 0, 0        # all zeros: every subarray sums to 0
    elif regime == 3:
        n = rng.randint(0, 10)
        vlo, vhi = -5, 5
    else:
        n = rng.randint(0, 14)
        vlo, vhi = -1, 1       # small alphabet, many collisions

    a = [rng.randint(vlo, vhi) for _ in range(n)]

    # Choose S so that hits are common: sometimes an actual subarray sum,
    # sometimes 0, sometimes random (possibly unreachable).
    pick = rng.randint(0, 3)
    if pick == 0 and n > 0:
        l = rng.randint(0, n - 1)
        r = rng.randint(l, n - 1)
        S = sum(a[l:r+1])
    elif pick == 1:
        S = 0
    elif pick == 2:
        S = rng.randint(-10, 10)
    else:
        S = rng.randint(-30, 30)

    out = [f"{n} {S}"]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
