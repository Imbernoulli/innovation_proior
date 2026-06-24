import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 7)
    # allow up to a fair number of edges so multi-edges / self-loops appear
    m = random.randint(0, 10)
    print(n, m)
    for _ in range(m):
        a = random.randint(1, n)
        b = random.randint(1, n)
        # bias a little toward self-loops and multi-edges by sometimes reusing a
        if random.random() < 0.15:
            b = a  # self-loop
        print(a, b)

main()
