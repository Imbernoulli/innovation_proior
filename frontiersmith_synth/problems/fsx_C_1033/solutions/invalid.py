# TIER: invalid
# Emits non-finite garbage for every rate -- must be rejected by the checker's
# finiteness check and score 0.
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    print(" ".join(["nan"] * T))


if __name__ == "__main__":
    main()
