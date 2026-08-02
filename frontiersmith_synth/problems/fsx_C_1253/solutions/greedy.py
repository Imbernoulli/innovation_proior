# TIER: greedy
"""The textbook "capacity-optimal" cache-blocking recipe: pick a single cube
tile size T that just fills the cache (3 tiles of T*T words <= capacity C),
clipped into the legal [min_t, max_t] range, canonical i,j,k loop order, and
NO padding. This is the first thing any competent programmer reaches for --
it accounts for capacity but is blind to cache SET geometry (associativity /
conflict aliasing) and to prefetch-friendly access order."""
import sys
import math


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    C = int(data[1]); L = int(data[2]); A = int(data[3])
    # PAD_MAX = int(data[4])  # ignored -- greedy never pads

    max_t = max(1, N // 3)
    min_t = min(2, max_t)
    T = min(max_t, max(min_t, int(math.isqrt(C // 3))))

    print(T, T, T)
    print(0, 0, 0)
    print("ijk")


if __name__ == "__main__":
    main()
