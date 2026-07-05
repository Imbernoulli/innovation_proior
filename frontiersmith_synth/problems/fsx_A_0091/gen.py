import sys, random

def main():
    i = int(sys.argv[1])
    rng = random.Random(9100 + i)
    # difficulty ladder: grid side m grows 6 -> 15
    m = 5 + i                      # i=1..10 -> m=6..15
    # embargo fraction on windows t>=1 cycles 0.08, 0.18, 0.28
    frac = 0.08 + 0.10 * ((i - 1) % 3)

    # candidate embargo pool = all slots with t >= 1 (never embargo window 0)
    pool = [(w, t) for w in range(m) for t in range(1, m)]
    e = int(frac * m * m)
    e = min(e, len(pool))
    emb = rng.sample(pool, e)
    emb.sort()

    out = [str(m), str(len(emb))]
    out.extend("%d %d" % (w, t) for (w, t) in emb)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
