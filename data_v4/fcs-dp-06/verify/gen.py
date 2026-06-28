import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes so the SOS structure gets exercised:
    #  - small bit width (lots of collisions / disjointness)
    #  - full 20-bit width
    #  - sparse masks (few bits set -> many disjoint pairs)
    #  - lots of zeros (stresses the self-pair correction)
    mode = rng.randint(0, 4)

    if mode == 0:
        n = rng.randint(0, 12)
        bits = rng.randint(0, 4)
    elif mode == 1:
        n = rng.randint(0, 40)
        bits = rng.randint(0, 6)
    elif mode == 2:
        n = rng.randint(0, 60)
        bits = 20            # full width, values < 2^20
    elif mode == 3:
        # sparse: each value has at most 2 bits set within `bits`
        n = rng.randint(0, 50)
        bits = rng.randint(1, 20)
    else:
        # zero-heavy
        n = rng.randint(0, 40)
        bits = rng.randint(1, 8)

    hi = (1 << bits)
    vals = []
    for _ in range(n):
        if mode == 3:
            k = rng.randint(0, 2)
            v = 0
            for _ in range(k):
                v |= (1 << rng.randint(0, bits - 1))
            vals.append(v)
        elif mode == 4:
            if rng.random() < 0.5:
                vals.append(0)
            else:
                vals.append(rng.randint(0, hi - 1))
        else:
            vals.append(rng.randint(0, hi - 1))

    out = [str(n)]
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
