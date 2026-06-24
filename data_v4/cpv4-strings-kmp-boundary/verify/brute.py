import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    n = len(s)

    count = 0
    sumTile = 0
    # For each prefix length L, directly test every candidate tile length d.
    # The prefix s[0:L] is "tiled" iff there exists d with 1 <= d < L,
    # d divides L, and s[0:L] equals (s[0:d]) repeated (L//d) times.
    # We take the SMALLEST such d as the minimal tile length.
    for L in range(1, n + 1):
        prefix = s[:L]
        found = None
        for d in range(1, L):           # strictly less than L: need >= 2 copies
            if L % d != 0:
                continue
            tile = s[:d]
            if tile * (L // d) == prefix:
                found = d
                break                   # smallest d, since d increases
        if found is not None:
            count += 1
            sumTile += found

    print(f"{count} {sumTile}")

main()
