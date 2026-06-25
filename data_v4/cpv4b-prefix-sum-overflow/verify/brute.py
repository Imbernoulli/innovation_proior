import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Independent O(n^2) brute force: try every contiguous subarray, sum it, compare to S.
    count = 0
    for i in range(n):
        running = 0
        for j in range(i, n):
            running += a[j]
            if running == S:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
