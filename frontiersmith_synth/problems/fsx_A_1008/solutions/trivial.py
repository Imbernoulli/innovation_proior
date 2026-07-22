# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    k = int(next(it))
    sigma = int(next(it))
    Lmax = int(next(it))
    # ignore the rest of the input (automata tables) -- this baseline never looks
    # at the transition structure at all: just spam one fixed letter.
    letter = "0"
    length = min(Lmax, 60)
    print(letter * length)


if __name__ == "__main__":
    main()
