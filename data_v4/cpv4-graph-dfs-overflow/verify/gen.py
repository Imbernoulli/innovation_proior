import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 8)
    # node 1 is root with parent -1; each other node i picks a parent < i
    lines = []
    # assign a random permutation-ish so parents come earlier; node i's parent in 1..i-1
    # root is node 1
    print(n)
    for i in range(1, n + 1):
        if i == 1:
            p = -1
        else:
            p = random.randint(1, i - 1)
        c = random.randint(0, 6)
        print(p, c)

main()
