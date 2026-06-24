import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small ranges so brute force is cheap and corner cases appear often.
    # Deliberately include negatives, zeros, the empty pattern (m=0), m=1,
    # m>n, all-negative arrays, and small value alphabets so that resonances
    # (matches up to an additive shift) actually occur.
    n = rng.randint(0, 8)

    # Pick a value-generation mode to stress different sign regimes.
    mode = rng.randint(0, 4)
    def val():
        if mode == 0:   # tiny alphabet incl 0 and negatives -> many shifts collide
            return rng.randint(-2, 2)
        elif mode == 1: # all non-positive
            return rng.randint(-5, 0)
        elif mode == 2: # all negative
            return rng.randint(-6, -1)
        elif mode == 3: # zeros heavy
            return rng.choice([0, 0, 0, 1, -1])
        else:           # wider
            return rng.randint(-9, 9)

    t = [val() for _ in range(n)]

    # Pattern length: allow 0 (empty), 1, up to n+1 (so m>n happens).
    m = rng.randint(0, max(2, n + 1))

    # Half the time, embed a shifted copy of a text window into the pattern so
    # that a real resonance exists (exercises the matching path, not just misses).
    p = []
    if m >= 1 and n >= m and rng.random() < 0.5:
        i = rng.randint(0, n - m)
        c = rng.randint(-4, 4)
        p = [t[i + j] - c for j in range(m)]
    else:
        p = [val() for _ in range(m)]

    out = []
    out.append(str(n))
    out.append(" ".join(str(x) for x in t))
    out.append(str(m))
    out.append(" ".join(str(x) for x in p))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
