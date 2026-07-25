import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # Small n so the brute force is fast; include n in {0,1} sometimes.
    r = random.random()
    if r < 0.05:
        n = 0
    elif r < 0.12:
        n = 1
    else:
        n = random.randint(2, 9)
    out = [str(n)]
    # For each node v >= 2, choose a parent p with 1 <= p < v.
    # Mixing "small parent" (deep chains) and "random parent" (bushy) shapes.
    for v in range(2, n + 1):
        if random.random() < 0.4:
            p = v - 1            # chain-ish: forces depth, stresses overflow path
        else:
            p = random.randint(1, v - 1)
        out.append(str(p))
    print("\n".join(out))

main()
