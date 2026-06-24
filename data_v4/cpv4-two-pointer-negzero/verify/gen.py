import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress the negative/zero/all-negative/empty corners.
    regime = rng.randint(0, 6)
    if regime == 0:
        n = 0
    elif regime == 1:
        n = 1
    else:
        n = rng.randint(2, 8)

    # Value pool intentionally centered on 0 and able to go all-negative or all-zero.
    pools = [
        lambda: rng.randint(-6, 6),     # mixed signs incl. zero
        lambda: rng.randint(-6, -1),    # all negative
        lambda: rng.choice([0, 0, 0, 3, -3]),  # lots of zeros
        lambda: rng.randint(0, 6),      # nonnegative incl. zero
    ]
    pick = rng.choice(pools)
    a = [pick() for _ in range(n)]

    # Threshold can be negative, zero, or positive, and can sit outside the reachable range.
    T = rng.randint(-12, 12)

    out = [str(n), str(T)]
    out.extend(str(x) for x in a)
    print(" ".join(out))

main()
