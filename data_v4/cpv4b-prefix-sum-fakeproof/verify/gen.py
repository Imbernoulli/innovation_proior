import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes so the parity bookkeeping and the popcount-vs-value distinction
    # both get exercised: empty arrays, tiny values, and values whose low bit and
    # popcount parity disagree (the trap zone).
    mode = seed % 5
    if mode == 0:
        n = rng.randint(0, 2)
        hi = 7
    elif mode == 1:
        n = rng.randint(1, 10)
        hi = 3                      # values 0..3: popcount parity and low bit often clash
    elif mode == 2:
        n = rng.randint(1, 12)
        hi = (1 << 10) - 1
    elif mode == 3:
        n = rng.randint(1, 12)
        hi = (1 << 30) - 1          # wide values near the stated bound
    else:
        n = rng.randint(1, 14)
        # force many odd-popcount, even-value and even-popcount, odd-value items
        choices = [1, 2, 3, 5, 6, 7, 0]
        a = [rng.choice(choices) for _ in range(n)]
        print(n)
        print(" ".join(map(str, a)))
        return

    a = [rng.randint(0, hi) for _ in range(n)]
    print(n)
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
