import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to exercise correctness AND the overflow corner:
    #  - small coords (catch logic bugs, collinear triples, duplicates)
    #  - large coords near 1e9 (where an int cross-product would overflow)
    mode = seed % 4
    n = rng.randint(0, 8)

    if mode == 0:
        lo, hi = -5, 5            # tiny: many collinear / duplicate triples
    elif mode == 1:
        lo, hi = -50, 50          # small
    elif mode == 2:
        lo, hi = -10**9, 10**9    # full magnitude: overflow regime
    else:
        # extreme corners only: coordinates pinned to +-1e9
        corners = [-10**9, 10**9]
        out = [str(n)]
        for _ in range(n):
            out.append(f"{rng.choice(corners)} {rng.choice(corners)}")
        print("\n".join(out))
        return

    out = [str(n)]
    for _ in range(n):
        out.append(f"{rng.randint(lo, hi)} {rng.randint(lo, hi)}")
    print("\n".join(out))

if __name__ == "__main__":
    main()
