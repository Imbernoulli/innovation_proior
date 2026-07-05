# TIER: invalid
# Emits a dense contiguous block of channels; pairwise sums collide immediately
# (e.g. 1+4 == 2+3), so it is NOT a Sidon set -> checker scores 0.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0])
    m = min(n, 60)
    print(" ".join(str(c) for c in range(1, m + 1)))

if __name__ == "__main__":
    main()
