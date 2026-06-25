import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 9)
    # Bias toward small magnitudes with lots of negatives and zeros so the
    # all-negative / sign / base-case corners are exercised frequently.
    mode = seed % 5
    vals = []
    for _ in range(n):
        if mode == 0:
            vals.append(random.randint(-6, -1))      # all negative
        elif mode == 1:
            vals.append(random.choice([-3, -2, -1, 0, 0, 1, 2, 3]))
        elif mode == 2:
            vals.append(random.randint(-9, 9))
        elif mode == 3:
            vals.append(random.choice([-1, 0, 1]))   # tiny magnitudes
        else:
            vals.append(random.randint(-100, 100))
    print(n)
    print(" ".join(map(str, vals)))

if __name__ == "__main__":
    main()
