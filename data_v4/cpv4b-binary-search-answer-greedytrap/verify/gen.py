import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # small cases so the assignment brute force stays fast
    n = random.randint(0, 9)
    if n == 0:
        m = random.randint(1, 4)
        print(n, m)
        return
    m = random.randint(1, min(n, 4))

    # mix of value regimes to exercise the greedy trap
    regime = random.randint(0, 3)
    vals = []
    for _ in range(n):
        if regime == 0:
            vals.append(random.randint(1, 9))        # tiny
        elif regime == 1:
            vals.append(random.randint(1, 100))      # small
        elif regime == 2:
            vals.append(random.randint(1, 10**9))    # large (overflow regime)
        else:
            # clustered near a few values -> good for LPT counterexamples
            base = random.choice([5, 6, 7, 8])
            vals.append(base + random.randint(0, 2))

    print(n, m)
    print(*vals)

main()
