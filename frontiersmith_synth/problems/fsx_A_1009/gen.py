import sys, random


def bits_for(n):
    # min bits so that 0..n-1 are representable (n>=2)
    return max(1, (n - 1).bit_length())


def build_pairs(rng, n, bits, K, trap):
    """Return K distinct sorted pairs (a,b), a<b<n.
    trap=True: as many pairs as available (up to ~2/3 of K) are bit-complement pairs
    w.r.t. the 'bits'-width mask, so they collide onto the SAME OR-fingerprint under a
    plain counting code (every ladder step's (n, bits) is chosen so this availability
    comfortably covers the target -- see the plan table below)."""
    mask = (1 << bits) - 1
    pairs = set()

    if trap:
        cand = [a for a in range(n) if 0 <= (mask - a) < n and (mask - a) != a]
        rng.shuffle(cand)
        target_trap = max(1, (K * 2) // 3)
        for a in cand:
            if len(pairs) >= target_trap:
                break
            b = mask - a
            lo, hi = (a, b) if a < b else (b, a)
            pairs.add((lo, hi))

    # fill remainder with uniformly random distinct pairs
    guard = 0
    while len(pairs) < K and guard < 200000:
        guard += 1
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        pairs.add((lo, hi))

    return sorted(pairs)[:K]


def main():
    i = int(sys.argv[1])
    rng = random.Random(190400 + 97 * i)

    # difficulty ladder: n and K grow; trap density rises; budget slack T stays tight
    plan = {
        1:  dict(n=18,  K=4,  T=7, trap=False),
        2:  dict(n=24,  K=6,  T=7, trap=False),
        3:  dict(n=30,  K=8,  T=6, trap=False),
        4:  dict(n=40,  K=12, T=6, trap=True),
        5:  dict(n=52,  K=14, T=6, trap=True),
        6:  dict(n=96,  K=18, T=6, trap=True),
        7:  dict(n=90,  K=22, T=7, trap=True),
        8:  dict(n=120, K=28, T=7, trap=True),
        9:  dict(n=160, K=36, T=8, trap=True),
        10: dict(n=210, K=46, T=8, trap=True),
    }[i]

    n, K, T, trap = plan["n"], plan["K"], plan["T"], plan["trap"]
    bits = bits_for(n)
    m = bits + T
    cap = max((n + 1) // 2 + 5, 10)
    cap = min(cap, n)

    pairs = build_pairs(rng, n, bits, K, trap)
    K = len(pairs)  # in case fewer distinct pairs were reachable

    out = [f"{n} {m} {cap}", f"{K}"]
    for a, b in pairs:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
