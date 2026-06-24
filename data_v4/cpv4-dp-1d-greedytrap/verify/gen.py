import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep n tiny so the exponential, non-memoized brute force stays fast.
    n = rng.randint(0, 14)

    # Mix of regimes to exercise the greedy trap and the corners:
    #  - small random tolls (lots of ties)
    #  - adversarial alternating cheap/expensive to bait "jump to cheaper next"
    #  - occasional large magnitudes to stress 64-bit accumulation
    mode = rng.randint(0, 3)
    vals = []
    for i in range(n):
        if mode == 0:
            vals.append(rng.randint(0, 5))
        elif mode == 1:
            vals.append(rng.choice([1, 1, 100, 100]))
        elif mode == 2:
            # alternate small / big to bait local greedy
            vals.append(rng.randint(0, 2) if i % 2 == 0 else rng.randint(50, 100))
        else:
            vals.append(rng.randint(0, 1000000000))

    out = [str(n)]
    out.append(" ".join(str(v) for v in vals))
    sys.stdout.write("\n".join(out) + "\n")

main()
