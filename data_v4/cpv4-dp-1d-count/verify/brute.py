import sys

def solve(n, K, MOD):
    # Brute force: enumerate every tiling of a 1xn strip with tiles of length 1 and 2,
    # assign each tile one of K colors, no two adjacent tiles share a color.
    # Count valid colorings, mod MOD.
    # We enumerate all "compositions" of n into parts 1 and 2 (the tile-length sequence),
    # then for each composition count the colorings directly.
    count = 0

    def gen_compositions(rem):
        # yield lists of 1s and 2s summing to rem
        if rem == 0:
            yield []
            return
        if rem >= 1:
            for rest in gen_compositions(rem - 1):
                yield [1] + rest
        if rem >= 2:
            for rest in gen_compositions(rem - 2):
                yield [2] + rest

    for comp in gen_compositions(n):
        t = len(comp)  # number of tiles
        if t == 0:
            # empty strip (n==0): exactly one tiling (no tiles)
            count += 1
            continue
        # colorings: first tile K choices, each subsequent tile (K-1) choices
        ways = K
        for _ in range(t - 1):
            ways *= (K - 1)
        count += ways

    return count % MOD


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); K = int(data[1]); MOD = int(data[2])
    print(solve(n, K, MOD))


if __name__ == "__main__":
    main()
