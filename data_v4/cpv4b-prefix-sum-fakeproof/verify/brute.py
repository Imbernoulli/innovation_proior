import sys


def popcount(x):
    return bin(x).count("1")


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Independent O(n^2) brute force: enumerate every window [l, r], maintain its
    # running XOR, and count it when the XOR has an even number of set bits.
    cnt = 0
    for l in range(n):
        x = 0
        for r in range(l, n):
            x ^= a[r]
            if popcount(x) % 2 == 0:
                cnt += 1
    print(cnt)


if __name__ == "__main__":
    main()
