import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    w = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Brute force: enumerate every contiguous subarray [i..j], sum it
    # directly, and count those with sum <= S. O(n^2) but obviously correct.
    count = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += w[j]
            if s <= S:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
