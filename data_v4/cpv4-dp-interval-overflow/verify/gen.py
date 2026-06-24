import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # tiny cases: n in [0, 7], labels small enough that brute is fast but big
    # enough to make products meaningful. Occasionally use larger labels to
    # exercise the magnitude (still tiny n so brute is instant).
    n = rng.randint(0, 7)
    big = rng.random() < 0.3
    hi = 2000 if big else 12
    lo = 1 if not big else 1500
    vals = [rng.randint(lo, hi) for _ in range(n)]
    out = [str(n)]
    out.append(' '.join(map(str, vals)))
    sys.stdout.write('\n'.join(out) + '\n')

if __name__ == "__main__":
    main()
