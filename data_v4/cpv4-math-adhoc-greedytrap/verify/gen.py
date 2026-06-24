import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # Small N so the O(N*sqrt N) brute force is fast, but cover the tricky residues.
    # Mix uniform small values with deliberately nasty ones (4^k*(8m+7), squares, etc.).
    pick = random.random()
    if pick < 0.6:
        n = random.randint(1, 3000)
    elif pick < 0.75:
        # force a 4^k*(8m+7) form -> answer 4
        k = random.randint(0, 3)
        m = random.randint(0, 50)
        n = (4 ** k) * (8 * m + 7)
        n = max(1, min(n, 20000))
    elif pick < 0.9:
        # force a perfect square -> answer 1
        r = random.randint(1, 140)
        n = r * r
    else:
        n = random.randint(1, 20000)
    print(n)

main()
