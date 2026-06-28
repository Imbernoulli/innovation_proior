import sys, random

# Random test generator for the rod-cutting problem.
# Usage: python3 gen.py <seed> [mode]
# Prints a valid stdin instance: n, then n prices p[1..n] (one per line here,
# whitespace-separated is also accepted by the solution).

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "tiny":
        n = rng.randint(0, 8)
    elif mode == "small":
        n = rng.randint(1, 14)
    elif mode == "mid":
        n = rng.randint(1, 18)
    else:
        n = rng.randint(1, 16)

    prices = []
    style = rng.randint(0, 4)
    for k in range(1, n + 1):
        if style == 0:
            # generic random prices
            prices.append(rng.randint(0, 30))
        elif style == 1:
            # roughly increasing -> tempts "cut nothing / few big pieces"
            prices.append(rng.randint(0, 5) + k * rng.randint(0, 3))
        elif style == 2:
            # roughly decreasing -> tempts "many length-1 pieces"
            prices.append(rng.randint(0, 5) + (n - k + 1) * rng.randint(0, 3))
        elif style == 3:
            # zeros and spikes
            prices.append(0 if rng.random() < 0.5 else rng.randint(1, 40))
        else:
            # bias toward making best price-per-length greedy wrong:
            # give a high per-length on a long piece but an even better split
            prices.append(rng.randint(1, 20))

    out = [str(n)]
    out.extend(str(x) for x in prices)
    sys.stdout.write(" ".join(out) + "\n")


main()
