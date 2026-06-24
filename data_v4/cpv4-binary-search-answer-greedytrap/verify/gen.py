import random, sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # tiny cases so the exhaustive DP brute stays fast
    n = random.randint(0, 9)
    if n == 0:
        k = random.randint(1, 4)
        print(n, k)
        print()
        return
    k = random.randint(1, max(1, n + 1))   # allow k > n sometimes
    # mix of value distributions to expose the greedy trap:
    style = random.randint(0, 3)
    vals = []
    for _ in range(n):
        if style == 0:
            vals.append(random.randint(0, 9))          # small, with zeros
        elif style == 1:
            vals.append(random.randint(1, 20))         # moderate
        elif style == 2:
            vals.append(random.choice([1, 1, 1, 50]))  # spiky: one big among smalls
        else:
            vals.append(random.randint(0, 100))        # wide range
    print(n, k)
    print(' '.join(map(str, vals)))

if __name__ == "__main__":
    main()
