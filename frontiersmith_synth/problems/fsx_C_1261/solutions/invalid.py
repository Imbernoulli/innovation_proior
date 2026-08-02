# TIER: invalid
import sys

# Stays idle the entire horizon regardless of the jobs -- since every
# case has jobs with strictly positive work, this always leaves work
# unfinished at some deadline and must be rejected before scoring.


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    print(" ".join("0" for _ in range(T)))


if __name__ == "__main__":
    main()
