# TIER: invalid
# Emits the all-zero table: not a permutation (every value repeats), so the
# feasibility check rejects it and the score is 0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    N = 1 << n
    sys.stdout.write("\n".join("0" for _ in range(N)) + "\n")

if __name__ == "__main__":
    main()
