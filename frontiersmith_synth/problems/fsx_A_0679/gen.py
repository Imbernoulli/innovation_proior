import sys

# difficulty ladder testId 1..10: grid dims R,C grow from 5x5 up to 9x9,
# alternating aspect ratio so both square and rectangular grids appear.
SIZES = [
    (5, 5), (5, 6), (6, 5), (6, 6), (6, 7),
    (7, 6), (7, 7), (7, 8), (8, 8), (9, 9),
]
PRIMES = [1000000007, 998244353]


def main():
    t = int(sys.argv[1])
    idx = (t - 1) % len(SIZES)
    R, C = SIZES[idx]
    p = PRIMES[(t - 1) % len(PRIMES)]
    print(R, C, p)


if __name__ == "__main__":
    main()
